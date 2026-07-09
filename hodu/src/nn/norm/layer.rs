//! Layer normalization with a learned affine (`gamma`, `beta`) over the last axis.
use hodu_core::{Ctx, Error, Tensor};

use crate::nn::{Module, Param};

/// `gamma * (x - mean) / sqrt(var + eps) + beta`, normalized over the last axis.
pub struct LayerNorm {
    gamma: Param,
    beta: Param,
    eps: f32,
}

impl LayerNorm {
    /// `size` = the normalized (last) dimension. `gamma` inits to 1, `beta` to 0.
    pub fn new(ctx: &Ctx, size: usize, eps: f32) -> LayerNorm {
        LayerNorm {
            gamma: Param::new(ctx, vec![1.0; size], vec![size]),
            beta: Param::new(ctx, vec![0.0; size], vec![size]),
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let axis = x.rank().saturating_sub(1);
        let norm = x.layernorm(axis, self.eps)?;
        let scaled = norm.try_mul(self.gamma.tensor())?; // (.., D) * (D,) broadcasts
        scaled.try_add(self.beta.tensor())
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_zero_mean_unit_var() {
        // gamma=1, beta=0 -> each row normalized to ~zero mean, unit variance.
        let ctx = Ctx::cpu();
        let ln = LayerNorm::new(&ctx, 4, 1e-5);
        let x = ctx.constant(vec![1., 2., 3., 4.], vec![1, 4]);
        let y = ln.forward(&x).unwrap().realize();
        let mean = y.iter().sum::<f32>() / 4.0;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var {var}");
    }
}
