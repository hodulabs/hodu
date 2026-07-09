//! Multi-head self-attention over kurumi's fused `sdpa` (+ optional RoPE).
use hodu_core::{Ctx, Error, Tensor};

use super::{Linear, Module};

/// Multi-head self-attention. Projects `x` `[B, S, d]` with four `Linear(d, d)`
/// maps (q/k/v/out), splits q/k/v into `H` heads of width `dh = d/H` and moves them
/// to `[B, H, S, dh]`, optionally applies RoPE to q and k, runs scaled dot-product
/// attention (the `1/sqrt(dh)` scale and the optional causal mask live inside
/// kurumi's `sdpa`), then merges the heads back to `[B, S, d]` and applies the
/// output projection. Autograd flows through `sdpa`/`rope` via the engine's VJP.
pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    causal: bool,
    use_rope: bool,
}

impl MultiHeadAttention {
    /// `d_model` must be divisible by `n_heads`. `causal` masks future keys;
    /// `use_rope` adds rotary position embedding to q/k. `seed` seeds the four
    /// projections (deterministically decorrelated).
    pub fn new(
        ctx: &Ctx,
        d_model: usize,
        n_heads: usize,
        causal: bool,
        use_rope: bool,
        seed: u64,
    ) -> Result<MultiHeadAttention, Error> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(Error::Shape {
                op: "MultiHeadAttention",
                msg: format!("d_model {d_model} not divisible by n_heads {n_heads}"),
            });
        }
        Ok(MultiHeadAttention {
            q: Linear::new(ctx, d_model, d_model, seed),
            k: Linear::new(ctx, d_model, d_model, seed ^ 0x1111),
            v: Linear::new(ctx, d_model, d_model, seed ^ 0x2222),
            o: Linear::new(ctx, d_model, d_model, seed ^ 0x3333),
            n_heads,
            causal,
            use_rope,
        })
    }

    // [B, S, d] -> [B, H, S, dh]: split the channel axis into heads, then transpose
    // heads ahead of the sequence so each (B, H) pair is an independent attention.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let dh = d / self.n_heads;
        x.reshape(vec![b, s, self.n_heads, dh])?.permute(vec![0, 2, 1, 3])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut q = self.split_heads(&self.q.forward(x)?)?;
        let mut k = self.split_heads(&self.k.forward(x)?)?;
        let v = self.split_heads(&self.v.forward(x)?)?;
        if self.use_rope {
            q = q.rope()?; // rotate over [.., S, dh] per head
            k = k.rope()?;
        }
        let attn = q.sdpa(&k, &v, self.causal)?; // [B, H, S, dh]
        let merged = attn.permute(vec![0, 2, 1, 3])?.reshape(vec![b, s, d])?;
        self.o.forward(&merged)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("q".to_string(), &self.q as &dyn Module),
            ("k".to_string(), &self.k),
            ("v".to_string(), &self.v),
            ("o".to_string(), &self.o),
        ]
    }
}
