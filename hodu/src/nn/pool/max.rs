//! Max pooling, 1/2/3-D. Parameter-free spatial layers.
use crate::nn::Module;
use hodu_core::{Error, Tensor};

/// 1-D max pooling with window `k` and stride `s` (both scalars).
pub struct MaxPool1d {
    k: usize,
    s: usize,
}

impl MaxPool1d {
    pub fn new(k: usize, s: usize) -> MaxPool1d {
        MaxPool1d { k, s }
    }
}

impl Module for MaxPool1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.max_pool1d(self.k, self.s)
    }
}

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

/// 3-D max pooling with window `k` and stride `s` (both `(d, h, w)`).
pub struct MaxPool3d {
    k: (usize, usize, usize),
    s: (usize, usize, usize),
}

impl MaxPool3d {
    pub fn new(k: (usize, usize, usize), s: (usize, usize, usize)) -> MaxPool3d {
        MaxPool3d { k, s }
    }
}

impl Module for MaxPool3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.max_pool3d(self.k, self.s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hodu_core::Ctx;

    #[test]
    fn max_pool1d_forward_shape() {
        // length 6, window 2, stride 2 -> length 3.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 6], vec![1, 1, 6]);
        assert_eq!(MaxPool1d::new(2, 2).forward(&x).unwrap().shape(), &[1, 1, 3]);
    }

    #[test]
    fn max_pool3d_forward_shape() {
        // 4x4x4, window 2, stride 2 -> 2x2x2.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 4 * 4 * 4], vec![1, 1, 4, 4, 4]);
        assert_eq!(MaxPool3d::new((2, 2, 2), (2, 2, 2)).forward(&x).unwrap().shape(), &[1, 1, 2, 2, 2]);
    }
}
