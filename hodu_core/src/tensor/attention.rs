//! Sequence / attention primitives: embedding gather, rotary position embedding
//! (RoPE), and scaled dot-product attention -- thin wraps of kurumi's decomposed
//! ops, each of which autodiffs for free through the engine's VJP.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// Rotary position embedding over the trailing `[S, D]` (D even, positions
    /// `0..S`, base 10000): a norm-preserving per-position rotation, so attention
    /// scores become sensitive to relative offsets. Leading axes are batch/heads.
    pub fn rope(&self) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.rope(n))
    }

    /// Scaled dot-product attention. `self`/`k`/`v` are `[..batch.., S, dh]` -- every
    /// leading dim is batch, so `[B, H, S, dh]` runs `B*H` attentions. Scaling by
    /// `1/sqrt(dh)` and the optional causal mask are built into the engine op.
    /// Returns `[..batch.., S, dh]`.
    pub fn sdpa(&self, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor, Error> {
        let (q, kn, vn) = (self.node(), k.node(), v.node());
        self.ctx().build(|g| g.sdpa(q, kn, vn, causal))
    }
}
