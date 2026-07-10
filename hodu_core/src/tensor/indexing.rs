//! Indexing on `Tensor`: `gather` (jnp.take), a thin wrap of kurumi's decomposed
//! gather op (autodiffs for free through the engine's VJP). Mirrors the engine's
//! `core/indexing.rs`.
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
}
