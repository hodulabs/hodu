//! The v3 `.hodu` write path: `write_container` lays out the metadata region (header + meta +
//! tensor table + descriptors + graph blob) in memory, pads to the page boundary, then streams
//! the page-aligned data region (payloads at their per-tensor 64B-aligned offsets). The byte
//! codec + shared io helpers live in the parent `container.rs`.
use super::descriptor::write_descriptors;
use super::{Entry, MAGIC, REGION_ALIGN, TENSOR_ALIGN, VERSION, align_up, write_str};
use crate::nn::QuantDescriptor;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

// `descriptors` is the quant-descriptor table (empty for a non-quant model -> just a 4-byte
// zero count). It sits AFTER the tensor table and BEFORE the graph blob.
//
// `graph` is the optional runnable-graph blob (a `save_runnable` artifact), written as a
// `[u64 len][bytes]` section after the descriptor table (len 0 if none). It stays in the small
// metadata region -- only the tensor payloads move to the page-aligned data region.
pub(in crate::serialize) fn write_container(
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
    write_descriptors(&mut m, descriptors)?;
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
