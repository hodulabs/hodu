//! `Flatten`: the parameter-free layer that collapses trailing dims into one.
use hodu_core::{Error, Tensor};

use super::Module;

/// Flatten dims `[start_dim ..]` into one (e.g. `[N,C,H,W] -> [N, C*H*W]` for
/// `start_dim = 1`), to feed conv features into a Linear head.
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: usize) -> Flatten {
        Flatten { start_dim }
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        x.flatten(self.start_dim)
    }
}
