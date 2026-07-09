//! The `.hodu` container internals: the tensor-table row (`Entry`), the on-disk
//! codecs + io helpers, and the model<->rows mapping (`model_entries` /
//! `apply_to_model`). The public `save`/`load` API in the parent drives these.
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::nn::Module;

const MAGIC: &[u8; 4] = b"HODU";
const VERSION: u32 = 1;

// tensor kind tags
const K_PARAM: u8 = 0; // learnable weight
const K_BUFFER: u8 = 1; // non-learnable f32 state (e.g. BatchNorm running stats)
pub(super) const K_OPTIM: u8 = 2; // optimizer moment / step
const K_QBUFFER: u8 = 3; // non-learnable raw-byte state (e.g. a packed quant weight)

pub(super) const DT_F32: u8 = 0; // f32-LE payload
const DT_U8: u8 = 1; // raw u8 payload (packed quant weight)

// A decoded tensor-table row. `data` is the raw LE payload; `dtype` says how to read it.
pub(super) struct Entry {
    pub(super) name: String,
    pub(super) kind: u8,
    pub(super) dtype: u8,
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<u8>,
}

pub(super) fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub(super) fn meta() -> [(&'static str, &'static str); 2] {
    [("format", "hodu"), ("created_by", concat!("hodu ", env!("CARGO_PKG_VERSION")))]
}

fn kind_name(k: u8) -> &'static str {
    match k {
        K_PARAM => "param",
        K_BUFFER => "buffer",
        K_OPTIM => "optim",
        K_QBUFFER => "byte-buffer",
        _ => "unknown",
    }
}

pub(super) fn inval(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
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

// The model's params + buffers (f32) + byte-buffers (raw dtype) as named tensor rows
// (FQN naming, stable per arch).
pub(super) fn model_entries(model: &dyn Module) -> Vec<Entry> {
    let mut out = Vec::new();
    for (name, p) in model.named_parameters("") {
        out.push(Entry {
            name,
            kind: K_PARAM,
            dtype: DT_F32,
            shape: p.shape().to_vec(),
            data: f32_to_bytes(&p.value()),
        });
    }
    for (name, b) in model.named_buffers("") {
        out.push(Entry {
            name,
            kind: K_BUFFER,
            dtype: DT_F32,
            shape: b.shape().to_vec(),
            data: f32_to_bytes(&b.value()),
        });
    }
    for (name, b) in model.named_byte_buffers("") {
        out.push(Entry { name, kind: K_QBUFFER, dtype: DT_U8, shape: b.shape().to_vec(), data: b.bytes() });
    }
    out
}

// `graph` is the optional runnable-graph blob (a `save_runnable` artifact). It is written
// as a trailing `[u64 len][bytes]` section AFTER the tensor table; an empty blob writes
// nothing, so a weights-only file is byte-identical to a plain `save` and any reader that
// stops after the table (see `read_container`) ignores the section.
pub(super) fn write_container(
    path: impl AsRef<Path>,
    meta: &[(&str, &str)],
    tensors: &[Entry],
    graph: &[u8],
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(meta.len() as u32).to_le_bytes())?;
    for (k, v) in meta {
        write_str(&mut w, k)?;
        write_str(&mut w, v)?;
    }
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for t in tensors {
        write_str(&mut w, &t.name)?;
        w.write_all(&[t.kind, t.dtype])?;
        w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for &d in &t.shape {
            w.write_all(&(d as u64).to_le_bytes())?;
        }
        w.write_all(&(t.data.len() as u64).to_le_bytes())?;
        w.write_all(&t.data)?;
    }
    if !graph.is_empty() {
        w.write_all(&(graph.len() as u64).to_le_bytes())?;
        w.write_all(graph)?;
    }
    w.flush()
}

// Returns the tensor rows plus the trailing runnable-graph blob (empty for a weights-only
// file). The blob, if present, is a `[u64 len][bytes]` section after the last row.
pub(super) fn read_container(path: impl AsRef<Path>) -> io::Result<(Vec<Entry>, Vec<u8>)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(inval("not a .hodu file (bad magic)"));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(inval(format!("unsupported .hodu version {version} (this build reads v{VERSION})")));
    }
    // meta is extensible and nothing here is required yet -> read and skip.
    let meta_n = read_u32(&mut r)? as usize;
    for _ in 0..meta_n {
        read_str(&mut r)?;
        read_str(&mut r)?;
    }
    let n = read_u32(&mut r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(&mut r)?;
        let mut tags = [0u8; 2];
        r.read_exact(&mut tags)?;
        let (kind, dtype) = (tags[0], tags[1]);
        if dtype != DT_F32 && dtype != DT_U8 {
            return Err(inval(format!("tensor '{name}': unsupported dtype tag {dtype}")));
        }
        let rank = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(&mut r)? as usize);
        }
        let nbytes = read_u64(&mut r)? as usize;
        let mut data = vec![0u8; nbytes];
        r.read_exact(&mut data)?;
        out.push(Entry { name, kind, dtype, shape, data });
    }
    // trailing runnable-graph section, if any (see write_container). A weights-only file
    // ends at the last row, so there is no trailer and the blob is empty.
    let mut trailer = Vec::new();
    r.read_to_end(&mut trailer)?;
    let graph = if trailer.is_empty() {
        Vec::new()
    } else if trailer.len() >= 8 {
        let len = u64::from_le_bytes(trailer[..8].try_into().unwrap()) as usize;
        if trailer.len() < 8 + len {
            return Err(inval("truncated graph section in .hodu file"));
        }
        trailer[8..8 + len].to_vec()
    } else {
        return Err(inval("truncated graph section in .hodu file"));
    };
    Ok((out, graph))
}

// Find a model tensor by (kind, name) among the file rows, validate its shape, and
// mark it consumed. O(n^2) linear scan -- fine for model tensor counts; swap for a
// name->index map if a file ever holds thousands of tensors.
fn take(
    entries: &[Entry],
    used: &mut [bool],
    kind: u8,
    name: &str,
    want: &[usize],
    want_dtype: u8,
) -> io::Result<Vec<u8>> {
    for (i, e) in entries.iter().enumerate() {
        if !used[i] && e.kind == kind && e.name == name {
            if e.shape.as_slice() != want {
                return Err(inval(format!("tensor '{name}' shape {:?} != model {want:?}", e.shape)));
            }
            if e.dtype != want_dtype {
                return Err(inval(format!("tensor '{name}' dtype {} != model dtype {want_dtype}", e.dtype)));
            }
            used[i] = true;
            return Ok(e.data.clone());
        }
    }
    Err(inval(format!("model {} '{name}' is missing from the .hodu file", kind_name(kind))))
}

// Populate the live model's params + buffers by name; error on any missing or extra
// (non-optim) tensor. optim rows are left for the caller (load_checkpoint) to apply.
pub(super) fn apply_to_model(entries: &[Entry], model: &dyn Module) -> io::Result<()> {
    let mut used = vec![false; entries.len()];
    for (name, p) in model.named_parameters("") {
        let bytes = take(entries, &mut used, K_PARAM, &name, p.shape(), DT_F32)?;
        p.set(bytes_to_f32(&bytes));
    }
    for (name, b) in model.named_buffers("") {
        let bytes = take(entries, &mut used, K_BUFFER, &name, b.shape(), DT_F32)?;
        b.set(bytes_to_f32(&bytes));
    }
    for (name, b) in model.named_byte_buffers("") {
        let bytes = take(entries, &mut used, K_QBUFFER, &name, b.shape(), DT_U8)?;
        b.set_bytes(bytes);
    }
    for (i, e) in entries.iter().enumerate() {
        if !used[i] && e.kind != K_OPTIM {
            return Err(inval(format!(
                "the .hodu file has {} '{}' with no match in the model",
                kind_name(e.kind),
                e.name
            )));
        }
    }
    Ok(())
}
