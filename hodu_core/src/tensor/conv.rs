//! 2-D convolution, a thin wrap of kurumi's `conv2d` op (the engine decomposes it
//! from strided-slice + dot_general, so autodiff -- conv backward and the weight
//! gradient -- comes for free). Layout is NCHW throughout.
use crate::Tensor;
use kurumi::Error;

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
}
