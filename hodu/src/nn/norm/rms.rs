//! RMS normalization over the last axis with a learned scale (the Llama-family norm).
use hodu_core::{Ctx, Error, Tensor};

use crate::nn::{Module, Param};

/// RMS normalization over the last axis with a learned scale (`gamma`, no bias):
/// `gamma * x / sqrt(mean(x^2) + eps)`. The norm Llama-family models use.
pub struct RmsNorm {
    gamma: Param,
    eps: f32,
}

impl RmsNorm {
    /// `size` = the normalized (last) dimension; `gamma` inits to 1.
    pub fn new(ctx: &Ctx, size: usize, eps: f32) -> RmsNorm {
        RmsNorm { gamma: Param::new(ctx, vec![1.0; size], vec![size]), eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let axis = x.rank().saturating_sub(1);
        let (n, eps) = (x.node(), self.eps);
        let norm = x.ctx().build(|g| g.rmsnorm(n, axis, eps))?;
        norm.try_mul(self.gamma.tensor()) // (.., D) * (D,) broadcasts
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_rms() {
        // gamma=1 -> row scaled to unit RMS (sqrt(mean(x^2)) == 1).
        let ctx = Ctx::cpu();
        let rn = RmsNorm::new(&ctx, 4, 1e-6);
        let x = ctx.constant(vec![1., 2., 3., 4.], vec![1, 4]);
        let y = rn.forward(&x).unwrap().realize();
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms {rms}");
    }
}
