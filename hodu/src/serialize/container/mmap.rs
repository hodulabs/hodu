//! The mmap-backed zero-copy reader for the v3 `.hodu` container. `MmapModel` maps the file,
//! parses only the metadata, and hands out `&[u8]` views into the page-aligned data region so a
//! large model's payloads page in on demand instead of being read whole. The byte codec + the
//! eager reader live in the parent `container.rs`.
use crate::nn::QuantDescriptor;
use crate::serialize::container::{Entry, REGION_ALIGN, TensorMeta, align_up, read_meta};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Cursor, Seek};
use std::path::Path;

/// A `.hodu` file memory-mapped for zero-copy weight access: the metadata is parsed eagerly
/// (small) and each tensor payload stays in the mapped, page-aligned data region. [`bytes`]
/// hands out a zero-copy `&[u8]` view into the mapping; pages fault in on demand, so a large
/// model is never read whole. The mapping is held for the model's lifetime -- a returned slice
/// borrows `self`. Drives [`load_mmap`](crate::serialize::load_mmap).
///
/// [`bytes`]: MmapModel::bytes
pub struct MmapModel {
    // SAFETY invariant: `mmap` outlives every `&[u8]` borrow handed out by `bytes`.
    mmap: Mmap,
    metas: Vec<TensorMeta>,
    region_base: usize,
    descriptors: Vec<QuantDescriptor>,
}

impl MmapModel {
    /// Memory-map `path` and parse its metadata region. Does NOT read the payloads -- they page
    /// in lazily when [`bytes`](MmapModel::bytes) touches them.
    pub fn open(path: impl AsRef<Path>) -> io::Result<MmapModel> {
        let file = File::open(path)?;
        // SAFETY: a read-only shared map of a plain file; we never mutate through it. A concurrent
        // external truncation could fault, which is the standard mmap caveat for on-disk artifacts.
        let mmap = unsafe { Mmap::map(&file)? };
        let (metas, descriptors, region_base) = {
            let mut cur = Cursor::new(&mmap[..]);
            let (metas, descriptors, _graph) = read_meta(&mut cur)?;
            (metas, descriptors, align_up(cur.stream_position()?, REGION_ALIGN) as usize)
        };
        Ok(MmapModel { mmap, metas, region_base, descriptors })
    }

    /// The file offset of the (4K-aligned) data-region base.
    pub fn region_offset(&self) -> u64 {
        self.region_base as u64
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.metas.len()
    }

    /// Whether the file has no tensors.
    pub fn is_empty(&self) -> bool {
        self.metas.is_empty()
    }

    /// Zero-copy view of tensor `i`'s payload -- a slice straight into the memory map.
    pub fn bytes(&self, i: usize) -> &[u8] {
        let tm = &self.metas[i];
        let start = self.region_base + tm.offset as usize;
        &self.mmap[start..start + tm.nbytes as usize]
    }

    pub(in crate::serialize) fn descriptors(&self) -> &[QuantDescriptor] {
        &self.descriptors
    }

    // Owned rows built by copying each payload out of the map (copy-on-apply). The file was never
    // read whole; only the metadata + the copied subset fault in.
    pub(in crate::serialize) fn entries(&self) -> Vec<Entry> {
        self.metas
            .iter()
            .enumerate()
            .map(|(i, tm)| Entry {
                name: tm.name.clone(),
                kind: tm.kind,
                dtype: tm.dtype,
                shape: tm.shape.clone(),
                data: self.bytes(i).to_vec(),
            })
            .collect()
    }
}
