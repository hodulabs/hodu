//! `.hodu` v1: a self-describing, named-tensor container. Header (magic + version +
//! ASCII KV meta) then one flat TENSOR TABLE: per entry a FQN name, a kind tag
//! (param | buffer | optim | byte-buffer), a dtype tag (f32 | u8), rank + dims, and a
//! raw LE payload sized by dtype (f32 = 4B, u8 = 1B). `load` populates the live model
//! BY NAME -- validating each shape/dtype and erroring on any missing / extra /
//! mismatched tensor -- instead of relying on `parameters()` order. Byte-buffers carry
//! a `QuantLinear`'s packed U8 weight at its real (small) size.
//!
//! This persists BatchNorm running stats (buffers), so eval-mode inference is correct
//! after a round-trip; `save_checkpoint` additionally stores optimizer state (moments
//! + step) for training resume. std + memmap2 (frontend-only) for the mmap reader, no serde.
//!
//! v3 (container.rs) splits the small metadata region from a 4K-page-aligned DATA REGION so the
//! file is mmap-able: [`load`] reads eagerly (back-compat), [`load_mmap`] maps the file and
//! copies weights out of the page-aligned region on demand -- a large model is never read whole.

mod container;
mod model;
mod runnable;
mod safetensors;

pub use container::MmapModel;
pub use runnable::{RunnableModel, load_runnable, save_multi, save_runnable};
pub use safetensors::{apply_safetensors, load_safetensors};

use crate::nn::Module;
use crate::optim::{OptState, SchedState};
use container::{DT_F32, Entry, K_OPTIM, bytes_to_f32, f32_to_bytes, inval, meta, read_container, write_container};
use model::{apply_to_model, model_entries};
use std::io;
use std::path::Path;

/// Write a model's params + buffers (named, self-describing) to `path`, plus a quant-descriptor
/// table so any `QuantLinear`'s scheme (bits/group_size/symmetric) is self-describing.
pub fn save(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    write_container(path, &meta(), &model_entries(model), &model.quant_descriptors(""), &[])
}

/// Load params + buffers from `path` into `model` by name. Errors on bad
/// magic/version, an unknown dtype, or any missing / extra / shape-mismatched tensor.
/// Optimizer rows (from a checkpoint) are ignored -- only the model state is applied.
pub fn load(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    let (entries, descriptors, _) = read_container(path)?;
    apply_to_model(&entries, &descriptors, model)
}

/// Load params + buffers into `model` by memory-mapping `path` instead of reading it whole: the
/// metadata is parsed eagerly (small) and each tensor is copied out of the page-aligned, mmap'd
/// data region on demand -- so a large model's file is paged in lazily, not slurped into RAM.
/// Same result as [`load`]; the mapping is dropped when this returns. For zero-copy `&[u8]` views
/// held for the model's lifetime, use [`MmapModel`] directly.
pub fn load_mmap(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    let m = MmapModel::open(path)?;
    apply_to_model(&m.entries(), m.descriptors(), model)
}

// K_OPTIM row name for a scheduler's resumable epoch counter -- a cross-frontend
// contract with hodu-py, so it must stay exactly this string.
const SCHED_ROW: &str = "sched.last_epoch";

/// Write model (params + buffers) AND optimizer state (moments + step) so a training run
/// can resume. Pass a `sched` to also persist its epoch counter (the `sched.last_epoch`
/// row). Load with [`load_checkpoint`]; plain [`load`] still reads the model.
pub fn save_checkpoint(
    path: impl AsRef<Path>,
    model: &dyn Module,
    opt: &dyn OptState,
    sched: Option<&dyn SchedState>,
) -> io::Result<()> {
    let mut entries = model_entries(model);
    for (name, data) in opt.state_dict() {
        entries.push(Entry { name, kind: K_OPTIM, dtype: DT_F32, shape: vec![data.len()], data: f32_to_bytes(&data) });
    }
    if let Some(s) = sched {
        let data = vec![s.last_epoch() as f32];
        entries.push(Entry {
            name: SCHED_ROW.to_string(),
            kind: K_OPTIM,
            dtype: DT_F32,
            shape: vec![1],
            data: f32_to_bytes(&data),
        });
    }
    write_container(path, &meta(), &entries, &model.quant_descriptors(""), &[])
}

/// Restore model AND optimizer from a checkpoint written by [`save_checkpoint`], so a run
/// resumes with moments + step intact. Pass a `sched` to also restore its epoch counter
/// from the `sched.last_epoch` row (Errs if a scheduler is given but the row is absent).
pub fn load_checkpoint(
    path: impl AsRef<Path>,
    model: &dyn Module,
    opt: &mut dyn OptState,
    sched: Option<&mut dyn SchedState>,
) -> io::Result<()> {
    let (entries, descriptors, _) = read_container(path)?;
    apply_to_model(&entries, &descriptors, model)?;
    let optim_sd: Vec<(String, Vec<f32>)> =
        entries.iter().filter(|e| e.kind == K_OPTIM).map(|e| (e.name.clone(), bytes_to_f32(&e.data))).collect();
    opt.load_state_dict(&optim_sd).map_err(|e| inval(format!("{e:?}")))?;
    if let Some(s) = sched {
        let row = entries
            .iter()
            .find(|e| e.kind == K_OPTIM && e.name == SCHED_ROW)
            .ok_or_else(|| inval(format!("checkpoint has no '{SCHED_ROW}' row for the scheduler")))?;
        s.set_last_epoch(bytes_to_f32(&row.data)[0] as usize);
    }
    Ok(())
}
