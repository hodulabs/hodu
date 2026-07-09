//! Parameter-free spatial layers: 2-D max and average pooling.
use hodu_core::{Error, Tensor};

use super::Module;

/// 2-D max pooling with window `k` and stride `s` (both `(h, w)`).
pub struct MaxPool2d {
    k: (usize, usize),
    s: (usize, usize),
}

impl MaxPool2d {
    pub fn new(k: (usize, usize), s: (usize, usize)) -> MaxPool2d {
        MaxPool2d { k, s }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.max_pool2d(self.k, self.s)
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

#[cfg(test)]
mod tests {
    use hodu_core::Ctx;

    use super::*;

    #[test]
    fn avg_pool2d_halves_and_averages() {
        // [1,1,2,2] of [1,2,3,4], 2x2 window -> mean 2.5, shape [1,1,1,1].
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]);
        let y = AvgPool2d::new((2, 2), (2, 2)).forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert!((y.realize()[0] - 2.5).abs() < 1e-6);
    }
}
