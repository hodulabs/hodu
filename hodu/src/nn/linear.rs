//! Affine layer `y = x @ w + b`, with `w: (in, out)` and `b: (out,)`.
use hodu_core::{Ctx, Error, Tensor};

use super::{Module, Param, kaiming_normal, uniform, xavier_uniform};

/// Weight initializer for [`Linear::with_init`]. `new` uses `HeUniform`.
pub enum Init {
    HeUniform,
    XavierUniform,
    KaimingNormal,
}

pub struct Linear {
    w: Param,
    b: Param,
}

impl Linear {
    /// He-uniform init in `[-1/sqrt(in), 1/sqrt(in)]` from a deterministic `seed`.
    pub fn new(ctx: &Ctx, in_features: usize, out_features: usize, seed: u64) -> Linear {
        Linear::with_init(ctx, in_features, out_features, seed, Init::HeUniform)
    }

    /// Same, with a chosen weight initializer.
    pub fn with_init(ctx: &Ctx, in_f: usize, out_f: usize, seed: u64, init: Init) -> Linear {
        let n = in_f * out_f;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (in_f as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, in_f, out_f, seed),
            Init::KaimingNormal => kaiming_normal(n, in_f, seed),
        };
        Linear { w: Param::new(ctx, w, vec![in_f, out_f]), b: Param::new(ctx, vec![0.0; out_f], vec![out_f]) }
    }

    /// The weight param, shape `[in, out]` (row-major). Used by `QuantLinear::from_linear`.
    pub fn weight(&self) -> &Param {
        &self.w
    }

    /// The bias param, shape `[out]`.
    pub fn bias(&self) -> &Param {
        &self.b
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let xw = x.matmul(self.w.tensor())?;
        xw.try_add(self.b.tensor()) // (N, out) + (out,) broadcasts
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}
