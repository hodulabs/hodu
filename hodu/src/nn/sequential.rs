//! A linear stack of modules: `forward` chains them; the tensor walks derive from
//! `children` (each layer named by its index).
use hodu_core::{Error, Tensor};

use super::Module;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let mut y = x.clone();
        for layer in &self.layers {
            y = layer.forward(&y)?;
        }
        Ok(y)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.layers.iter().enumerate().map(|(i, l)| (i.to_string(), l.as_ref())).collect()
    }
}
