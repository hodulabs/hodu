//! The shared per-channel affine used by the normalization layers; each norm
//! (layer / RMS / group / instance) lives in its own submodule.
use hodu_core::{Error, Tensor};

use super::Param;

mod group;
mod instance;
mod layer;
mod rms;

pub use group::GroupNorm;
pub use instance::InstanceNorm;
pub use layer::LayerNorm;
pub use rms::RmsNorm;

// per-channel affine over `[N, C, ..]`: gamma/beta live as [C], reshaped to
// [1, C, 1, ..] at forward so they broadcast on the channel axis (dim 1).
pub(super) fn channel_affine(x: &Tensor, gamma: &Param, beta: &Param) -> Result<Tensor, Error> {
    let mut sh = vec![1usize; x.rank()];
    sh[1] = gamma.shape()[0];
    let g = gamma.tensor().reshape(sh.clone())?;
    let b = beta.tensor().reshape(sh)?;
    x.try_mul(&g)?.try_add(&b)
}
