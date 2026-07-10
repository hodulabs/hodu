//! Normalization ops on `Tensor` (the raw engine op; the learned scale/shift lives
//! in `hodu::nn`).
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// Layer normalization over `axis` (mean 0, var 1; no affine -- see
    /// `hodu::nn::LayerNorm` for the learned scale/shift).
    pub fn layernorm(&self, axis: usize, eps: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.layernorm(n, axis, eps))
    }

    /// RMS normalization over `axis`: `x / sqrt(mean(x^2) + eps)` (no affine).
    pub fn rmsnorm(&self, axis: usize, eps: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.rmsnorm(n, axis, eps))
    }

    /// Group normalization over `[N, C, ..]`: split C into `groups`, normalize each (no affine).
    pub fn group_norm(&self, groups: usize, eps: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.group_norm(n, groups, eps))
    }

    /// Instance normalization over `[N, C, ..]` (group norm, one group per channel; no affine).
    pub fn instance_norm(&self, eps: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.instance_norm(n, eps))
    }
}
