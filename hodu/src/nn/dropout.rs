//! Inverted dropout for the static / build-once graph.
//!
//! A build-once graph can't add/remove nodes per step, so dropout is one
//! fixed subgraph driven by two fed values -- a shared train/eval `flag` and a
//! per-dropout `seed`. Train: mask ~= `Bernoulli(1-p)`, survivors scaled `1/(1-p)`;
//! eval (`flag=0`): threshold 0 keeps everything, scale 1 -> identity. `ctx.tick_rng()`
//! refeeds the seed each step for a fresh mask; `ctx.set_training(false)` flips to eval.
use crate::nn::Module;
use hodu_core::{Ctx, DType, Error, Tensor};

pub struct Dropout {
    seed: Tensor,
    p: f32,
    inv: f32, // 1/(1-p), the survivor scale in train mode
}

impl Dropout {
    /// Drop each element with probability `p` (must be in `[0, 1)`) in train mode.
    /// Registers a per-step seed with `ctx`; call `ctx.tick_rng()` each step for fresh
    /// masks. Errs on `p` out of range (returns `Result` like the other fallible layer
    /// constructors, e.g. `MultiHeadAttention::new`).
    pub fn new(ctx: &Ctx, p: f32) -> Result<Dropout, Error> {
        // p>=1 -> inv=inf (silent NaN in train); p<0 is nonsense. Reject both.
        if !(0.0..1.0).contains(&p) {
            return Err(Error::Shape { op: "Dropout::new", msg: format!("dropout p must be in [0,1), got {p}") });
        }
        Ok(Dropout { seed: ctx.new_dropout_seed(), p, inv: 1.0 / (1.0 - p) })
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let ctx = x.ctx();
        let flag = ctx.train_flag(); // [1]: 1.0 train, 0.0 eval
        let thresh = &flag * self.p; // train: p, eval: 0
        let scale = &(&flag * (self.inv - 1.0)) + 1.0; // train: 1/(1-p), eval: 1
        let shape = x.shape().to_vec();
        let (seed, thr) = (self.seed.node(), thresh.node());
        let mask = ctx.build(|g| {
            let u = g.rand_uniform_keyed(shape.clone(), seed)?; // fresh per step
            let tb = g.broadcast_to(thr, shape.clone())?;
            let keep = g.ge(u, tb)?; // survive where u >= thresh
            Ok(g.cast(keep, DType::F32))
        })?;
        Ok(&(x * &mask) * &scale)
    }
}
