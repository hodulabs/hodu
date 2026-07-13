//! The `.hodu` container byte format: the header, the tensor-table row (`Entry`), the io
//! helpers, and write/read_container. The model<->rows mapping lives in `model.rs`; the
//! public `save`/`load` API in the parent drives both.
//!
//! v3 layout (mmap-able): the METADATA region (header + meta + tensor table + descriptors +
//! graph blob) is small and read eagerly; the tensor table stores `(data_offset, nbytes)` into
//! a page-aligned DATA REGION (the concatenated payloads) instead of inline bytes. So a reader
//! can mmap the file and expose each tensor as a zero-copy `&[u8]` view (see [`MmapModel`]),
//! paged in on demand -- a large model is never read whole.
//!
//! ```text
//! MAGIC "HODU" + VERSION(u32=3) + meta(KV)
//! + n_tensors(u32) + per-tensor[ name, kind(u8), dtype(u8), rank(u32)+dims(u64), data_offset(u64), nbytes(u64) ]
//! + n_descriptors(u32) + descriptors
//! + graph_blob: u64 len + bytes (0 len if none)
//! + PAD to the next 4096-byte boundary
//! + DATA REGION: payloads at their data_offset (region base 4K-aligned; each tensor 64B-aligned)
//! ```
//! All little-endian. `data_offset` is relative to the DATA REGION start. The region base is
//! 4K-aligned (the mmap page unit) and each tensor is 64B-aligned within it, so a zero-copy view
//! is SIMD-friendly at a small padding cost.
mod mmap;

pub use mmap::MmapModel;

use crate::nn::QuantDescriptor;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"HODU";
const VERSION: u32 = 3; // v3 splits metadata from a page-aligned, mmap-able data region

// The mmap page unit the data region base is aligned to, and the per-tensor alignment within it.
pub(super) const REGION_ALIGN: u64 = 4096;
const TENSOR_ALIGN: u64 = 64;

// tensor kind tags (consumed by model.rs when it maps a Module to/from rows)
pub(super) const K_PARAM: u8 = 0; // learnable weight
pub(super) const K_BUFFER: u8 = 1; // non-learnable f32 state (e.g. BatchNorm running stats)
pub(super) const K_OPTIM: u8 = 2; // optimizer moment / step
pub(super) const K_QBUFFER: u8 = 3; // non-learnable raw-byte state (e.g. a packed quant weight)

// quant-descriptor scheme tags: 0 = affine group-wise (weight-only int4/int8), the only scheme
// today. Reserved: GPTQ/AWQ/MX/codebook get their own tag when added.
const SCHEME_AFFINE: u8 = 0;

pub(super) const DT_F32: u8 = 0; // f32-LE payload
pub(super) const DT_U8: u8 = 1; // raw u8 payload (packed quant weight)

// A decoded tensor-table row. `data` is the raw LE payload; `dtype` says how to read it.
pub(super) struct Entry {
    pub(super) name: String,
    pub(super) kind: u8,
    pub(super) dtype: u8,
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<u8>,
}

// A tensor-table row without its payload: the mmap/eager readers parse this from the metadata
// region, then locate the bytes at `region_base + offset` in the data region.
pub(super) struct TensorMeta {
    pub(super) name: String,
    pub(super) kind: u8,
    pub(super) dtype: u8,
    pub(super) shape: Vec<usize>,
    pub(super) offset: u64, // byte offset within the data region
    pub(super) nbytes: u64,
}

pub(super) fn meta() -> [(&'static str, &'static str); 2] {
    [("format", "hodu"), ("created_by", concat!("hodu ", env!("CARGO_PKG_VERSION")))]
}

pub(super) fn inval(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

pub(super) fn align_up(n: u64, a: u64) -> u64 {
    (n + a - 1) & !(a - 1) // a is a power of two (REGION_ALIGN / TENSOR_ALIGN)
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn read_str(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| inval(format!("bad utf8 in name: {e}")))
}

// f32 <-> LE bytes: the DT_F32 payload codec, shared by model.rs (params/buffers) and
// runnable.rs (constant weights).
pub(super) fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// `descriptors` is the quant-descriptor table (empty for a non-quant model -> just a 4-byte
// zero count). It sits AFTER the tensor table and BEFORE the graph blob.
//
// `graph` is the optional runnable-graph blob (a `save_runnable` artifact), written as a
// `[u64 len][bytes]` section after the descriptor table (len 0 if none). It stays in the small
// metadata region -- only the tensor payloads move to the page-aligned data region.
pub(super) fn write_container(
    path: impl AsRef<Path>,
    meta: &[(&str, &str)],
    tensors: &[Entry],
    descriptors: &[QuantDescriptor],
    graph: &[u8],
) -> io::Result<()> {
    // Assign each tensor a data-region offset: running sum of nbytes, 64B-aligned per tensor.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut off = 0u64;
    for t in tensors {
        off = align_up(off, TENSOR_ALIGN);
        offsets.push(off);
        off += t.data.len() as u64;
    }

    // Build the metadata region in memory (it is small by design), so its exact length -- and
    // thus the page-aligned data-region base -- is known before writing.
    let mut m: Vec<u8> = Vec::new();
    m.write_all(MAGIC)?;
    m.write_all(&VERSION.to_le_bytes())?;
    m.write_all(&(meta.len() as u32).to_le_bytes())?;
    for (k, v) in meta {
        write_str(&mut m, k)?;
        write_str(&mut m, v)?;
    }
    m.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (t, &o) in tensors.iter().zip(&offsets) {
        write_str(&mut m, &t.name)?;
        m.write_all(&[t.kind, t.dtype])?;
        m.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for &d in &t.shape {
            m.write_all(&(d as u64).to_le_bytes())?;
        }
        m.write_all(&o.to_le_bytes())?; // data_offset (into the data region)
        m.write_all(&(t.data.len() as u64).to_le_bytes())?; // nbytes
    }
    m.write_all(&(descriptors.len() as u32).to_le_bytes())?;
    for d in descriptors {
        write_str(&mut m, &d.weight_fqn)?;
        m.write_all(&[SCHEME_AFFINE, d.bits])?;
        m.write_all(&(d.group_size as u64).to_le_bytes())?;
        m.write_all(&[d.symmetric as u8])?;
        write_str(&mut m, &d.scales_fqn)?;
        write_str(&mut m, d.mins_fqn.as_deref().unwrap_or(""))?; // empty = none
    }
    m.write_all(&(graph.len() as u64).to_le_bytes())?; // always present (0 len if none)
    m.write_all(graph)?;

    let region_base = align_up(m.len() as u64, REGION_ALIGN);
    let pad = (region_base - m.len() as u64) as usize;

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&m)?;
    w.write_all(&vec![0u8; pad])?; // pad to the page boundary
    // Data region: payloads in offset order, zero-padding the intra-tensor alignment gaps.
    let mut written = 0u64;
    for (t, &o) in tensors.iter().zip(&offsets) {
        if o > written {
            w.write_all(&vec![0u8; (o - written) as usize])?;
        }
        w.write_all(&t.data)?;
        written = o + t.data.len() as u64;
    }
    w.flush()
}

// Parse the metadata region (magic..graph blob) from a seekable reader, leaving the cursor at
// the metadata end so the caller can compute the page-aligned data-region base. Shared by the
// eager reader and the mmap reader.
pub(super) fn read_meta(r: &mut (impl Read + Seek)) -> io::Result<(Vec<TensorMeta>, Vec<QuantDescriptor>, Vec<u8>)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(inval("not a .hodu file (bad magic)"));
    }
    let version = read_u32(r)?;
    if version != VERSION {
        return Err(inval(format!("unsupported .hodu version {version} (this build reads v{VERSION})")));
    }
    // meta is extensible and nothing here is required yet -> read and skip.
    let meta_n = read_u32(r)? as usize;
    for _ in 0..meta_n {
        read_str(r)?;
        read_str(r)?;
    }
    let n = read_u32(r)? as usize;
    let mut metas = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(r)?;
        let mut tags = [0u8; 2];
        r.read_exact(&mut tags)?;
        let (kind, dtype) = (tags[0], tags[1]);
        if dtype != DT_F32 && dtype != DT_U8 {
            return Err(inval(format!("tensor '{name}': unsupported dtype tag {dtype}")));
        }
        let rank = read_u32(r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(r)? as usize);
        }
        let offset = read_u64(r)?;
        let nbytes = read_u64(r)?;
        metas.push(TensorMeta { name, kind, dtype, shape, offset, nbytes });
    }
    // quant-descriptor table: 0 for a non-quant model.
    let nd = read_u32(r)? as usize;
    let mut descriptors = Vec::with_capacity(nd);
    for _ in 0..nd {
        let weight_fqn = read_str(r)?;
        let mut sb = [0u8; 2];
        r.read_exact(&mut sb)?;
        let (scheme, bits) = (sb[0], sb[1]);
        if scheme != SCHEME_AFFINE {
            return Err(inval(format!("quant descriptor '{weight_fqn}': unsupported scheme tag {scheme}")));
        }
        let group_size = read_u64(r)? as usize;
        let mut symb = [0u8; 1];
        r.read_exact(&mut symb)?;
        let scales_fqn = read_str(r)?;
        let mins = read_str(r)?;
        descriptors.push(QuantDescriptor {
            weight_fqn,
            bits,
            group_size,
            symmetric: symb[0] != 0,
            scales_fqn,
            mins_fqn: if mins.is_empty() { None } else { Some(mins) },
        });
    }
    // runnable-graph section: always a [u64 len][bytes] block (len 0 for a weights-only file).
    let glen = read_u64(r)? as usize;
    let mut graph = vec![0u8; glen];
    r.read_exact(&mut graph)?;
    Ok((metas, descriptors, graph))
}

// Returns the tensor rows (payloads read from the data region), the quant-descriptor table, and
// the runnable-graph blob (empty for a weights-only file). Eager path: reads only the metadata
// then the payloads (skipping alignment padding) -- the rows carry their bytes exactly as before,
// so apply_to_model / load_runnable are unchanged downstream.
pub(super) fn read_container(path: impl AsRef<Path>) -> io::Result<(Vec<Entry>, Vec<QuantDescriptor>, Vec<u8>)> {
    let mut r = BufReader::new(File::open(path)?);
    let (metas, descriptors, graph) = read_meta(&mut r)?;
    let region_base = align_up(r.stream_position()?, REGION_ALIGN);
    let mut out = Vec::with_capacity(metas.len());
    for tm in metas {
        r.seek(io::SeekFrom::Start(region_base + tm.offset))?;
        let mut data = vec![0u8; tm.nbytes as usize];
        r.read_exact(&mut data)?;
        out.push(Entry { name: tm.name, kind: tm.kind, dtype: tm.dtype, shape: tm.shape, data });
    }
    Ok((out, descriptors, graph))
}
