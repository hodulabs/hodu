//! Non-learnable, host-valued f32 Input leaf (e.g. BatchNorm running stats): like a
//! Param but excluded from parameters()/gradients -- a BUFFER the optimizer never
//! touches, updated host-side, and persisted by save/load so eval-mode state survives
//! a round-trip. Cheap-clone (`Rc`).
use hodu_core::{Ctx, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Buffer(Rc<BufferInner>);

struct BufferInner {
    ctx: Ctx,
    tensor: Tensor,
    shape: Vec<usize>,
    value: RefCell<Vec<f32>>,
}

impl Buffer {
    /// Create an Input leaf, seed its feed with `value`.
    pub fn new(ctx: &Ctx, value: Vec<f32>, shape: Vec<usize>) -> Buffer {
        let tensor = ctx.input(shape.clone());
        ctx.feed(tensor.node(), value.clone(), shape.clone());
        Buffer(Rc::new(BufferInner { ctx: ctx.clone(), tensor, shape, value: RefCell::new(value) }))
    }

    /// The graph handle for this buffer (use it in a forward pass).
    pub fn tensor(&self) -> &Tensor {
        &self.0.tensor
    }

    /// A copy of the current host value.
    pub fn value(&self) -> Vec<f32> {
        self.0.value.borrow().clone()
    }

    /// This buffer's shape.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Overwrite the host value and re-feed it (host-side update / `load`).
    pub fn set(&self, value: Vec<f32>) {
        *self.0.value.borrow_mut() = value;
        self.0.ctx.feed(self.0.tensor.node(), self.0.value.borrow().clone(), self.0.shape.clone());
    }
}
