//! Reductions on `Tensor`: sum/mean over one axis or all axes, max-reduce, and the
//! (non-differentiable) argmax/argmin index ops.
use crate::Tensor;
use kurumi::{Error, Graph, NodeId};

impl Tensor {
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.sum(n, axis))
    }
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.mean(n, axis))
    }
    /// Reduce over every axis to a scalar.
    pub fn sum_all(&self) -> Result<Tensor, Error> {
        let (n, r) = (self.node(), self.rank());
        self.ctx().build(|g| reduce_all(g, n, r, |g, x| g.sum(x, 0)))
    }
    /// Mean over every axis to a scalar.
    pub fn mean_all(&self) -> Result<Tensor, Error> {
        let (n, r) = (self.node(), self.rank());
        self.ctx().build(|g| reduce_all(g, n, r, |g, x| g.mean(x, 0)))
    }
    /// Max-reduce over `axis`.
    pub fn max_axis(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.reduce_max(n, axis))
    }
    /// Index (I64) of the max along `axis` (e.g. class predictions). Non-differentiable.
    pub fn argmax(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.argmax(n, axis))
    }
    /// Index (I64) of the min along `axis`. Non-differentiable.
    pub fn argmin(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.argmin(n, axis))
    }
}

fn reduce_all(
    g: &mut Graph,
    n: NodeId,
    rank: usize,
    step: impl Fn(&mut Graph, NodeId) -> Result<NodeId, Error>,
) -> Result<NodeId, Error> {
    let mut x = n;
    for _ in 0..rank {
        x = step(g, x)?;
    }
    Ok(x)
}
