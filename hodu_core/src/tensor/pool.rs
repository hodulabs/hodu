//! 2-D pooling on `Tensor`: max / average pool, thin wraps of kurumi's `pool` ops
//! (the engine decomposes them, so autodiff comes for free). Layout is NCHW.
//! Mirrors the engine's `nn/pool.rs`.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// 2-D max pool: window `k`, stride `s` (both `(h, w)`). `[N,C,H,W]` ->
    /// `[N,C,Ho,Wo]`.
    pub fn max_pool2d(&self, k: (usize, usize), s: (usize, usize)) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.max_pool2d(x, k, s))
    }

    /// 2-D average pool: window `k`, stride `s`.
    pub fn avg_pool2d(&self, k: (usize, usize), s: (usize, usize)) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.avg_pool2d(x, k, s))
    }
}
