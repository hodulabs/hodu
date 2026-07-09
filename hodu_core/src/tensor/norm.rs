//! Normalization ops on `Tensor` (the raw engine op; the learned scale/shift lives
//! in `hodu::nn`). rmsnorm/group_norm wraps land here as needed.
use kurumi::Error;

use crate::Tensor;

impl Tensor {
    /// Layer normalization over `axis` (mean 0, var 1; no affine -- see
    /// `hodu::nn::LayerNorm` for the learned scale/shift).
    pub fn layernorm(&self, axis: usize, eps: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.layernorm(n, axis, eps))
    }
}
