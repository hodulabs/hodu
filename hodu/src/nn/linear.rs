//! Affine layer `y = x @ w + b`, with `w: (in, out)` and `b: (out,)`.
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct Linear {
    w: Param,
    b: Option<Param>,
}

impl Linear {
    /// He-uniform init in `[-1/sqrt(in), 1/sqrt(in)]` from a deterministic `seed`.
    pub fn new(ctx: &Ctx, in_features: usize, out_features: usize, seed: u64) -> Linear {
        Linear::with_init(ctx, in_features, out_features, seed, Init::HeUniform, true)
    }

    /// Same, with a chosen weight initializer. `bias=false` drops the bias: no Param is
    /// allocated, `parameters()` omits it, and `forward` is the plain `x @ w`.
    pub fn with_init(ctx: &Ctx, in_f: usize, out_f: usize, seed: u64, init: Init, bias: bool) -> Linear {
        let n = in_f * out_f;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (in_f as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, in_f, out_f, seed),
            Init::KaimingNormal => kaiming_normal(n, in_f, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, in_f, out_f, seed),
        };
        let b = bias.then(|| Param::new(ctx, vec![0.0; out_f], vec![out_f]));
        Linear { w: Param::new(ctx, w, vec![in_f, out_f]), b }
    }

    /// The weight param, shape `[in, out]` (row-major). Used by `QuantLinear::from_linear`.
    pub fn weight(&self) -> &Param {
        &self.w
    }

    /// The bias param, shape `[out]`, or `None` when bias is disabled.
    pub fn bias(&self) -> Option<&Param> {
        self.b.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let xw = x.matmul(self.w.tensor())?;
        match &self.b {
            Some(b) => xw.try_add(b.tensor()), // (N, out) + (out,) broadcasts
            None => Ok(xw),
        }
    }
    fn parameters(&self) -> Vec<Param> {
        let mut ps = vec![self.w.clone()];
        ps.extend(self.b.clone());
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bias_drops_param_and_skips_offset() {
        let ctx = Ctx::cpu();
        let with = Linear::new(&ctx, 3, 2, 0);
        let without = Linear::with_init(&ctx, 3, 2, 0, Init::HeUniform, false);
        // no-bias layer has one fewer param, and no bias Param.
        assert_eq!(with.parameters().len(), 2);
        assert_eq!(without.parameters().len(), 1);
        assert!(without.bias().is_none());
        // forward is the plain matmul: no bias offset.
        let x = ctx.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = without.forward(&x).unwrap();
        let mm = x.matmul(without.weight().tensor()).unwrap();
        assert_eq!(ctx.eval_f32(y.node()), ctx.eval_f32(mm.node()));
    }
}
