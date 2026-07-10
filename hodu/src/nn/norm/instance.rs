//! Instance normalization (group norm with one group per channel).
use crate::nn::norm::channel_affine;
use crate::nn::{Module, Param};
use hodu_core::{Ctx, Error, Tensor};

/// Instance normalization: group norm with one group per channel.
pub struct InstanceNorm {
    gamma: Param,
    beta: Param,
    eps: f32,
}

impl InstanceNorm {
    pub fn new(ctx: &Ctx, channels: usize, eps: f32) -> InstanceNorm {
        InstanceNorm {
            gamma: Param::new(ctx, vec![1.0; channels], vec![channels]),
            beta: Param::new(ctx, vec![0.0; channels], vec![channels]),
            eps,
        }
    }
}

impl Module for InstanceNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let norm = x.instance_norm(self.eps)?;
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
    fn instancenorm_zero_mean_per_channel() {
        // [N=1, C=2, L=3]. Each channel normalized over its spatial dim -> ~0 mean.
        let ctx = Ctx::cpu();
        let inorm = InstanceNorm::new(&ctx, 2, 1e-5);
        let x = ctx.constant(vec![1., 2., 3., 10., 20., 30.], vec![1, 2, 3]);
        let y = inorm.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3]);
        let v = y.realize();
        let c0 = v[0..3].iter().sum::<f32>() / 3.0;
        let c1 = v[3..6].iter().sum::<f32>() / 3.0;
        assert!(c0.abs() < 1e-4, "ch0 mean {c0}");
        assert!(c1.abs() < 1e-4, "ch1 mean {c1}");
    }
}
