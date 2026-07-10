//! Parameter-free activation layers (unit structs wrapping a `Tensor` method).
use crate::nn::Module;
use hodu_core::{Error, Tensor};

/// ReLU: `max(0, x)`.
pub struct Relu;
/// Hyperbolic tangent.
pub struct Tanh;
/// Gaussian error linear unit.
pub struct Gelu;
/// Logistic sigmoid.
pub struct Sigmoid;

impl Module for Relu {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.relu())
    }
}
impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.tanh())
    }
}
impl Module for Gelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.gelu())
    }
}
impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.sigmoid())
    }
}
