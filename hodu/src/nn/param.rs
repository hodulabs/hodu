//! The learnable host-valued Input leaf. Wraps a fixed graph node plus its host value;
//! updates re-feed the node so the next eval sees them. Cheap-clone (`Rc`). The
//! non-learnable Buffer (f32) / QBuffer (raw-byte) variants live in the submodules.

mod buffer;
mod qbuffer;

pub use buffer::Buffer;
pub use qbuffer::QBuffer;

use hodu_core::{Ctx, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// A trainable parameter. Cheap-clone (`Rc`): the layer and the optimizer share
/// one, so an optimizer step is visible to the next forward.
#[derive(Clone)]
pub struct Param(Rc<ParamInner>);

struct ParamInner {
    ctx: Ctx,
    tensor: Tensor,
    shape: Vec<usize>,
    value: RefCell<Vec<f32>>,
}

impl Param {
    /// Create an Input leaf, seed its feed with `value`.
    pub fn new(ctx: &Ctx, value: Vec<f32>, shape: Vec<usize>) -> Param {
        let tensor = ctx.input(shape.clone());
        ctx.feed(tensor.node(), value.clone(), shape.clone());
        Param(Rc::new(ParamInner { ctx: ctx.clone(), tensor, shape, value: RefCell::new(value) }))
    }

    /// The graph handle for this param (use it in a forward pass).
    pub fn tensor(&self) -> &Tensor {
        &self.0.tensor
    }

    /// A copy of the current host value.
    pub fn value(&self) -> Vec<f32> {
        self.0.value.borrow().clone()
    }

    /// This param's shape.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Overwrite the host value and re-feed it (for `load_state_dict`).
    pub fn set(&self, value: Vec<f32>) {
        *self.0.value.borrow_mut() = value;
        self.0.ctx.feed(self.0.tensor.node(), self.0.value.borrow().clone(), self.0.shape.clone());
    }

    /// `value -= lr * grad`, then re-feed the new value for the next eval. The
    /// optimizer passes a pre-scaled delta with `lr = 1.0` when it owns the LR.
    pub fn apply_grad(&self, lr: f32, grad: &[f32]) {
        {
            let mut v = self.0.value.borrow_mut();
            for (vi, gi) in v.iter_mut().zip(grad) {
                *vi -= lr * gi;
            }
        }
        self.0.ctx.feed(self.0.tensor.node(), self.0.value.borrow().clone(), self.0.shape.clone());
    }
}
