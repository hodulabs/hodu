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
mod descriptor;
mod mmap;
mod read;
mod write;

pub use mmap::MmapModel;
pub(super) use read::{read_container, read_meta};
pub(super) use write::write_container;

use std::io::{self, Read, Write};

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
