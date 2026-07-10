//! A lazy tensor handle: an IR node in its `Ctx`'s graph, plus cached shape and
//! dtype. Binary ops broadcast + promote (NumPy rules) before the engine's strict
//! builders, so `&a + &b` and `x.matmul(&w) + b` behave like NumPy. Clone is an
//! `Rc` bump; shape/dtype are known at record time, so errors point at the user's
//! line, not an eval stack.
//!
//! This file is the spine: the handle and grad. The broadcasting machine (`bin` +
//! promote + coerce) lives in `tensor/broadcast.rs`, and the op surface is split by
//! domain across the other `tensor/*.rs`.

mod activation;
mod attention;
mod broadcast;
mod conv;
mod elementwise;
mod explog;
mod indexing;
mod linalg;
mod loss;
mod movement;
mod norm;
mod operators;
mod pool;
mod reduce;

use crate::Ctx;
use kurumi::{DType, Error, NodeId};
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor(Rc<TensorInner>);

struct TensorInner {
    ctx: Ctx,
    node: NodeId,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    pub(crate) fn new(ctx: Ctx, node: NodeId, shape: Vec<usize>, dtype: DType) -> Tensor {
        Tensor(Rc::new(TensorInner { ctx, node, shape, dtype }))
    }

    pub fn node(&self) -> NodeId {
        self.0.node
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }
    pub fn ctx(&self) -> &Ctx {
        &self.0.ctx
    }

    /// Realize to host f32, evaluated with the ctx's current feeds.
    pub fn realize(&self) -> Vec<f32> {
        self.0.ctx.eval_f32(self.0.node)
    }
    /// Realize and read the first element (for scalars / loss monitoring).
    pub fn item(&self) -> f32 {
        self.realize()[0]
    }

    /// A rank-1 f32 scalar in this tensor's ctx (broadcasts against anything).
    pub fn scalar_like(&self, v: f32) -> Tensor {
        self.0.ctx.constant(vec![v], vec![1])
    }

    /// Grad of `self` (a scalar loss) w.r.t. each `wrt` tensor. Returns grad
    /// handles that realize (with the current feeds) to the gradients.
    pub fn grad(&self, wrt: &[&Tensor]) -> Result<Vec<Tensor>, Error> {
        let ws: Vec<NodeId> = wrt.iter().map(|t| t.0.node).collect();
        let ids = self.0.ctx.grad(self.0.node, &ws)?;
        Ok(ids.into_iter().map(|id| self.0.ctx.build_inf(|_| id)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_matches_oracle() {
        let ctx = Ctx::cpu();
        let a = ctx.constant(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = ctx.constant(vec![5., 6., 7., 8.], vec![2, 2]);
        let y = (&a.matmul(&b).unwrap() + &a).relu();
        // [[19,22],[43,50]] + [[1,2],[3,4]] = [[20,24],[46,54]], relu = identity
        assert_eq!(y.realize(), vec![20., 24., 46., 54.]);
        assert_eq!(y.shape(), &[2, 2]);
    }

    #[test]
    fn broadcast_row_plus_bias() {
        // (2,3) + (3,) -> (2,3): the Linear bias case.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = ctx.constant(vec![10., 20., 30.], vec![3]);
        assert_eq!((&x + &b).realize(), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn scalar_ops_and_mean() {
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![1., 2., 3., 4.], vec![2, 2]);
        assert_eq!((&x * 2.0).realize(), vec![2., 4., 6., 8.]);
        assert_eq!(x.mean_all().unwrap().item(), 2.5);
    }

    #[test]
    fn record_errors_eagerly() {
        let ctx = Ctx::cpu();
        let a = ctx.constant(vec![1., 2.], vec![2]);
        let b = ctx.constant(vec![1., 2., 3.], vec![3]);
        assert!(a.try_add(&b).is_err()); // shape mismatch at record time
    }

    #[test]
    fn grad_of_square() {
        // d/dx sum(x^2) = 2x
        let ctx = Ctx::cpu();
        let x = ctx.input(vec![3]);
        ctx.feed(x.node(), vec![1., 2., 3.], vec![3]);
        let loss = x.square().sum_all().unwrap();
        let g = &loss.grad(&[&x]).unwrap()[0];
        assert_eq!(g.realize(), vec![2., 4., 6.]);
    }

    #[test]
    fn argmax_picks_max_index() {
        // rows [0.1,0.7,0.2] and [3,1,2] -> argmax axis 1 = [1, 0].
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.1, 0.7, 0.2, 3.0, 1.0, 2.0], vec![2, 3]);
        assert_eq!(x.argmax(1).unwrap().cast(DType::F32).realize(), vec![1.0, 0.0]);
    }

    #[test]
    fn argmin_picks_min_index() {
        // rows [0.1,0.7,0.2] and [3,1,2] -> argmin axis 1 = [0, 1].
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.1, 0.7, 0.2, 3.0, 1.0, 2.0], vec![2, 3]);
        assert_eq!(x.argmin(1).unwrap().cast(DType::F32).realize(), vec![0.0, 1.0]);
    }
}
