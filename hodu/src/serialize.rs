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
//! + step) for training resume. std only, no serde.
//!
//! Next step (hodu-plan/02-artifact-format.md): promote this flat table to
//! section-offset regions with 4K page alignment for mmap. The name/kind/dtype/shape
//! schema here is exactly the table that layout builds on.
use std::io;
use std::path::Path;

use crate::nn::Module;
use crate::optim::OptState;

mod container;
mod model;
mod runnable;

pub use runnable::save_runnable;

use container::{DT_F32, Entry, K_OPTIM, inval, meta, read_container, write_container};
use model::{apply_to_model, bytes_to_f32, f32_to_bytes, model_entries};

/// Write a model's params + buffers (named, self-describing) to `path`.
pub fn save(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    write_container(path, &meta(), &model_entries(model), &[])
}

/// Load params + buffers from `path` into `model` by name. Errors on bad
/// magic/version, an unknown dtype, or any missing / extra / shape-mismatched tensor.
/// Optimizer rows (from a checkpoint) are ignored -- only the model state is applied.
pub fn load(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    let (entries, _) = read_container(path)?;
    apply_to_model(&entries, model)
}

/// Write model (params + buffers) AND optimizer state (moments + step) so a training
/// run can resume. Load with [`load_checkpoint`]; plain [`load`] still reads the model.
pub fn save_checkpoint(path: impl AsRef<Path>, model: &dyn Module, opt: &dyn OptState) -> io::Result<()> {
    let mut entries = model_entries(model);
    for (name, data) in opt.state_dict() {
        entries.push(Entry { name, kind: K_OPTIM, dtype: DT_F32, shape: vec![data.len()], data: f32_to_bytes(&data) });
    }
    write_container(path, &meta(), &entries, &[])
}

/// Restore model AND optimizer from a checkpoint written by [`save_checkpoint`], so a
/// run resumes with moments + step intact.
pub fn load_checkpoint(path: impl AsRef<Path>, model: &dyn Module, opt: &mut dyn OptState) -> io::Result<()> {
    let (entries, _) = read_container(path)?;
    apply_to_model(&entries, model)?;
    let optim_sd: Vec<(String, Vec<f32>)> =
        entries.iter().filter(|e| e.kind == K_OPTIM).map(|e| (e.name.clone(), bytes_to_f32(&e.data))).collect();
    opt.load_state_dict(&optim_sd).map_err(|e| inval(format!("{e:?}")))
}
