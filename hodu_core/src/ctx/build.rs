//! Graph-record escape hatches: run a raw kurumi builder closure over the record
//! graph and wrap the resulting node(s) as `Tensor`(s), so an op not surfaced as a
//! `Tensor` method is still one call away. State lives in `CtxInner.graph` (see
//! ctx.rs).
use crate::{Ctx, Tensor};
use kurumi::{Error, Graph, NodeId};

impl Ctx {
    /// Escape hatch: record any kurumi op via the raw builder and wrap the
    /// result, so an op not surfaced as a `Tensor` method is still one call away:
    /// `ctx.build(|g| g.some_op(a.node(), ..))`. The borrow ends before `wrap`
    /// re-borrows (the `borrow_mut` temporary drops at statement end).
    pub fn build<F>(&self, f: F) -> Result<Tensor, Error>
    where
        F: FnOnce(&mut Graph) -> Result<NodeId, Error>,
    {
        let n = f(&mut self.0.graph.borrow_mut())?;
        Ok(self.wrap(n))
    }

    /// Same, for infallible builders (constant / unary ops).
    pub fn build_inf<F: FnOnce(&mut Graph) -> NodeId>(&self, f: F) -> Tensor {
        let n = f(&mut self.0.graph.borrow_mut());
        self.wrap(n)
    }

    /// Like [`Ctx::build`] for a builder that yields SEVERAL nodes (e.g. `g.split`);
    /// wraps each. The `borrow_mut` drops at statement end, before `wrap` re-borrows.
    pub fn build_many<F>(&self, f: F) -> Result<Vec<Tensor>, Error>
    where
        F: FnOnce(&mut Graph) -> Result<Vec<NodeId>, Error>,
    {
        let ns = f(&mut self.0.graph.borrow_mut())?;
        Ok(ns.into_iter().map(|n| self.wrap(n)).collect())
    }

    pub(super) fn wrap(&self, node: NodeId) -> Tensor {
        let g = self.0.graph.borrow();
        Tensor::new(self.clone(), node, g.shape(node).to_vec(), g.dtype(node))
    }
}
