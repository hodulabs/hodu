//! 2-D convolution and pooling, thin wraps of kurumi's `conv2d`/`pool` ops (the
//! engine decomposes them from strided-slice + dot_general, so autodiff -- conv
//! backward and the weight gradient -- comes for free). Layout is NCHW throughout.
use kurumi::Error;

use crate::Tensor;

impl Tensor {
    /// 2-D convolution. `self` `[N, C, H, W]`, `weight` `[O, C, KH, KW]` ->
    /// `[N, O, Ho, Wo]`. No bias (the layer adds it). `stride`/`padding`/`dilation`
    /// are `(h, w)` pairs.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv2d(x, w, stride, padding, dilation))
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
}
