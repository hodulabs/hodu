//! Sequence / attention primitives: embedding gather, rotary position embedding
//! (RoPE), and scaled dot-product attention -- thin wraps of kurumi's decomposed
//! ops, each of which autodiffs for free through the engine's VJP.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// Gather slices of `self` along `axis` at integer `indices` (jnp.take): the
    /// `axis` dim is replaced by all of `indices`' dims. Embedding lookup is
    /// `table.gather(idx, 0)` with `table` `[vocab, d]` and `idx` an int tensor
    /// `[..]` -> `[.., d]`. `indices` must be an integer tensor (see
    /// [`crate::Ctx::input_i64`]).
    pub fn gather(&self, indices: &Tensor, axis: usize) -> Result<Tensor, Error> {
        let (operand, idx) = (self.node(), indices.node());
        self.ctx().build(|g| g.gather(operand, idx, axis))
    }

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
