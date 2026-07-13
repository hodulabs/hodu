//! Token embedding: a `[vocab, d]` lookup table gathered by integer token ids.
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

/// Maps integer token ids to `d`-dim rows of a learned `[vocab, d]` table.
/// `forward(idx)` gathers rows, so an id tensor `idx` `[B, S]` -> `[B, S, d]`. `idx`
/// must be an integer tensor (from [`Ctx::input_i64`](hodu_core::Ctx::input_i64) or
/// an int constant); the table itself is an ordinary f32 param and trains normally.
pub struct Embedding {
    weight: Param,
}

impl Embedding {
    /// Uniform init in `[-1/sqrt(d), 1/sqrt(d)]` from a deterministic `seed`.
    pub fn new(ctx: &Ctx, vocab: usize, d: usize, seed: u64) -> Embedding {
        Embedding::with_init(ctx, vocab, d, seed, Init::HeUniform)
    }

    /// Same as [`Embedding::new`], with a chosen weight initializer (fan_in = vocab,
    /// fan_out = d for the Xavier/Kaiming variants).
    pub fn with_init(ctx: &Ctx, vocab: usize, d: usize, seed: u64, init: Init) -> Embedding {
        let n = vocab * d;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (d as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, vocab, d, seed),
            Init::KaimingNormal => kaiming_normal(n, vocab, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, vocab, d, seed),
        };
        Embedding { weight: Param::new(ctx, w, vec![vocab, d]) }
    }
}

impl Module for Embedding {
    /// `idx` is an integer id tensor; the returned rows carry a trailing `d` axis.
    fn forward(&self, idx: &Tensor) -> Result<Tensor, Error> {
        self.weight.tensor().gather(idx, 0)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_changes_weights() {
        let ctx = Ctx::cpu();
        let default = Embedding::new(&ctx, 6, 8, 7);
        let kaiming = Embedding::with_init(&ctx, 6, 8, 7, Init::KaimingNormal);
        assert_ne!(
            default.weight.value(),
            kaiming.weight.value(),
            "a non-default init must change the initial weights"
        );
        // new defaults to He-uniform: same seed + HeUniform reproduces `new` exactly.
        let he = Embedding::with_init(&ctx, 6, 8, 7, Init::HeUniform);
        assert_eq!(default.weight.value(), he.weight.value());
    }
}
