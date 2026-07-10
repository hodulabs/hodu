//! Pre-norm Transformer encoder block, and a stack of them.
use crate::nn::{LayerNorm, Linear, Module, MultiHeadAttention};
use hodu_core::{Ctx, Error, Tensor};

/// A pre-norm Transformer block: `x = x + MHA(LN1(x))`, then `x = x + FFN(LN2(x))`,
/// where `FFN = Linear(d, 4d) -> Gelu -> Linear(4d, d)`. Pre-norm (LayerNorm before
/// each sublayer, residual added raw) is the stable variant that trains without a
/// warmup schedule; the two residual paths keep gradients flowing to the embedding.
pub struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl TransformerBlock {
    /// `d_model`/`n_heads` size attention; `causal`/`use_rope` are passed through to
    /// the attention sublayer. The FFN hidden width is `4 * d_model`.
    pub fn new(
        ctx: &Ctx,
        d_model: usize,
        n_heads: usize,
        causal: bool,
        use_rope: bool,
        seed: u64,
    ) -> Result<TransformerBlock, Error> {
        Ok(TransformerBlock {
            ln1: LayerNorm::new(ctx, d_model, 1e-5),
            attn: MultiHeadAttention::new(ctx, d_model, n_heads, causal, use_rope, seed)?,
            ln2: LayerNorm::new(ctx, d_model, 1e-5),
            ff1: Linear::new(ctx, d_model, 4 * d_model, seed ^ 0xABCD),
            ff2: Linear::new(ctx, 4 * d_model, d_model, seed ^ 0xDCBA),
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let h = x.try_add(&self.attn.forward(&self.ln1.forward(x)?)?)?;
        let f = self.ff2.forward(&self.ff1.forward(&self.ln2.forward(&h)?)?.gelu())?;
        h.try_add(&f)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("ln1".to_string(), &self.ln1 as &dyn Module),
            ("attn".to_string(), &self.attn),
            ("ln2".to_string(), &self.ln2),
            ("ff1".to_string(), &self.ff1),
            ("ff2".to_string(), &self.ff2),
        ]
    }
}

/// A stack of `n_layers` `TransformerBlock`s applied in sequence (each with its own
/// params). `forward` chains them; `parameters` aggregates all layers.
pub struct TransformerEncoder {
    blocks: Vec<TransformerBlock>,
}

impl TransformerEncoder {
    pub fn new(
        ctx: &Ctx,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        causal: bool,
        use_rope: bool,
        seed: u64,
    ) -> Result<TransformerEncoder, Error> {
        let blocks = (0..n_layers)
            .map(|i| {
                TransformerBlock::new(
                    ctx,
                    d_model,
                    n_heads,
                    causal,
                    use_rope,
                    seed.wrapping_add(i as u64 * 0x9E37_79B9),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TransformerEncoder { blocks })
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let mut y = x.clone();
        for b in &self.blocks {
            y = b.forward(&y)?;
        }
        Ok(y)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.blocks.iter().enumerate().map(|(i, b)| (i.to_string(), b as &dyn Module)).collect()
    }
}
