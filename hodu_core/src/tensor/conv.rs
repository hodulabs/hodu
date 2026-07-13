//! Convolution & transposed convolution, thin wraps of kurumi's `conv*` ops (the engine
//! decomposes them from strided-slice + dot_general, so autodiff -- conv backward and the
//! weight gradient -- comes for free). Layout is N,C,(spatial...) throughout.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// 1-D convolution. `self` `[N, C, L]`, `weight` `[O, C, K]` -> `[N, O, Lo]`. No bias
    /// (the layer adds it). `stride`/`padding`/`dilation` are scalars.
    pub fn conv1d(&self, weight: &Tensor, stride: usize, padding: usize, dilation: usize) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv1d(x, w, stride, padding, dilation))
    }

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

    /// 3-D convolution. `self` `[N, C, D, H, W]`, `weight` `[O, C, Kd, Kh, Kw]` ->
    /// `[N, O, Do, Ho, Wo]`. No bias (the layer adds it). `stride`/`padding`/`dilation`
    /// are `(d, h, w)` triples.
    pub fn conv3d(
        &self,
        weight: &Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv3d(x, w, stride, padding, dilation))
    }

    /// 1-D transposed convolution. `self` `[N, C, L]`, `weight` `[C, O, K]` (in_ch first) ->
    /// `[N, O, Lo]`. No bias (the layer adds it). `stride`/`padding`/`output_padding`/`dilation`
    /// are scalars.
    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv_transpose1d(x, w, stride, padding, output_padding, dilation))
    }

    /// 2-D transposed convolution. `self` `[N, C, H, W]`, `weight` `[C, O, KH, KW]` (in_ch first)
    /// -> `[N, O, Ho, Wo]`. No bias (the layer adds it). `stride`/`padding`/`output_padding`/`dilation`
    /// are `(h, w)` pairs.
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv_transpose2d(x, w, stride, padding, output_padding, dilation))
    }

    /// 3-D transposed convolution. `self` `[N, C, D, H, W]`, `weight` `[C, O, Kd, Kh, Kw]`
    /// (in_ch first) -> `[N, O, Do, Ho, Wo]`. No bias (the layer adds it).
    /// `stride`/`padding`/`output_padding`/`dilation` are `(d, h, w)` triples.
    pub fn conv_transpose3d(
        &self,
        weight: &Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<Tensor, Error> {
        let (x, w) = (self.node(), weight.node());
        self.ctx().build(|g| g.conv_transpose3d(x, w, stride, padding, output_padding, dilation))
    }
}
