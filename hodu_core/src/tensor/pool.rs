//! Pooling on `Tensor`: max / average pool, thin wraps of kurumi's `pool` ops (the engine
//! decomposes them, so autodiff comes for free). Layout is N,C,(spatial...).
//! Mirrors the engine's `nn/pool.rs`.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// 1-D max pool: window `k`, stride `s` (both scalars). `[N,C,L]` -> `[N,C,Lo]`.
    pub fn max_pool1d(&self, k: usize, s: usize) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.max_pool1d(x, k, s))
    }

    /// 1-D average pool: window `k`, stride `s`.
    pub fn avg_pool1d(&self, k: usize, s: usize) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.avg_pool1d(x, k, s))
    }

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

    /// 3-D max pool: window `k`, stride `s` (both `(d, h, w)`). `[N,C,D,H,W]` ->
    /// `[N,C,Do,Ho,Wo]`.
    pub fn max_pool3d(&self, k: (usize, usize, usize), s: (usize, usize, usize)) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.max_pool3d(x, k, s))
    }

    /// 3-D average pool: window `k`, stride `s`.
    pub fn avg_pool3d(&self, k: (usize, usize, usize), s: (usize, usize, usize)) -> Result<Tensor, Error> {
        let x = self.node();
        self.ctx().build(|g| g.avg_pool3d(x, k, s))
    }
}
