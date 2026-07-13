//! The v3 `.hodu` read path: `read_meta` parses the small metadata region (header + tensor
//! table + descriptors + graph blob) from a seekable reader, and `read_container` follows it by
//! copying each payload out of the page-aligned data region (eager load). The mmap reader reuses
//! `read_meta`. The byte codec + shared io helpers live in the parent `container.rs`.
use super::descriptor::read_descriptors;
use super::{
    DT_F32, DT_U8, Entry, MAGIC, REGION_ALIGN, TensorMeta, VERSION, align_up, inval, read_str, read_u32, read_u64,
};
use crate::nn::QuantDescriptor;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

// Parse the metadata region (magic..graph blob) from a seekable reader, leaving the cursor at
// the metadata end so the caller can compute the page-aligned data-region base. Shared by the
// eager reader and the mmap reader.
pub(in crate::serialize) fn read_meta(
    r: &mut (impl Read + Seek),
) -> io::Result<(Vec<TensorMeta>, Vec<QuantDescriptor>, Vec<u8>)> {
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
    let descriptors = read_descriptors(r)?;
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
pub(in crate::serialize) fn read_container(
    path: impl AsRef<Path>,
) -> io::Result<(Vec<Entry>, Vec<QuantDescriptor>, Vec<u8>)> {
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
