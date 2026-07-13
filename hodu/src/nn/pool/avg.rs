//! Average pooling, 1/2/3-D. Parameter-free spatial layers.
use crate::nn::Module;
use hodu_core::{Error, Tensor};

/// 1-D average pooling with window `k` and stride `s` (both scalars).
pub struct AvgPool1d {
    k: usize,
    s: usize,
}

impl AvgPool1d {
    pub fn new(k: usize, s: usize) -> AvgPool1d {
        AvgPool1d { k, s }
    }
}

impl Module for AvgPool1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.avg_pool1d(self.k, self.s)
    }
}

/// 2-D average pooling with window `k` and stride `s` (both `(h, w)`).
pub struct AvgPool2d {
    k: (usize, usize),
    s: (usize, usize),
}

impl AvgPool2d {
    pub fn new(k: (usize, usize), s: (usize, usize)) -> AvgPool2d {
        AvgPool2d { k, s }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.avg_pool2d(self.k, self.s)
    }
}

/// 3-D average pooling with window `k` and stride `s` (both `(d, h, w)`).
pub struct AvgPool3d {
    k: (usize, usize, usize),
    s: (usize, usize, usize),
}

impl AvgPool3d {
    pub fn new(k: (usize, usize, usize), s: (usize, usize, usize)) -> AvgPool3d {
        AvgPool3d { k, s }
    }
}

impl Module for AvgPool3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.avg_pool3d(self.k, self.s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hodu_core::Ctx;

    #[test]
    fn avg_pool2d_halves_and_averages() {
        // [1,1,2,2] of [1,2,3,4], 2x2 window -> mean 2.5, shape [1,1,1,1].
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]);
        let y = AvgPool2d::new((2, 2), (2, 2)).forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert!((y.realize()[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn avg_pool1d_forward_shape() {
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 6], vec![1, 1, 6]);
        assert_eq!(AvgPool1d::new(2, 2).forward(&x).unwrap().shape(), &[1, 1, 3]);
    }

    #[test]
    fn avg_pool3d_forward_shape() {
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 4 * 4 * 4], vec![1, 1, 4, 4, 4]);
        assert_eq!(AvgPool3d::new((2, 2, 2), (2, 2, 2)).forward(&x).unwrap().shape(), &[1, 1, 2, 2, 2]);
    }
}
