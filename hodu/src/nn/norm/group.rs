//! Group normalization over `[N, C, ..]` (split C into groups, then a per-channel affine).
use crate::nn::norm::channel_affine;
use crate::nn::{Module, Param};
use hodu_core::{Ctx, Error, Tensor};

/// Group normalization over `[N, C, ..]`: split C into `groups`, normalize each
/// group, then a learned per-channel affine (`gamma`, `beta`).
pub struct GroupNorm {
    gamma: Param,
    beta: Param,
    groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn new(ctx: &Ctx, channels: usize, groups: usize, eps: f32) -> GroupNorm {
        GroupNorm {
            gamma: Param::new(ctx, vec![1.0; channels], vec![channels]),
            beta: Param::new(ctx, vec![0.0; channels], vec![channels]),
            groups,
            eps,
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let norm = x.group_norm(self.groups, self.eps)?;
        channel_affine(&norm, &self.gamma, &self.beta)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groupnorm_normalizes_each_group() {
        // [N=1, C=4, H=2, W=2], 2 groups. gamma=1/beta=0 -> group 0 (channels 0,1,
        // flat idx 0..8) is normalized to ~0 mean / unit var.
        let ctx = Ctx::cpu();
        let gn = GroupNorm::new(&ctx, 4, 2, 1e-5);
        let x = ctx.constant((0..16).map(|i| i as f32).collect(), vec![1, 4, 2, 2]);
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 2, 2]);
        let g0 = &y.realize()[0..8];
        let mean = g0.iter().sum::<f32>() / 8.0;
        let var = g0.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-4, "group mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "group var {var}");
    }
}
