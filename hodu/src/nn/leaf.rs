//! Host-valued Input leaves: `Param` (learnable), `Buffer` (non-learnable f32, e.g.
//! BatchNorm running stats), and `QBuffer` (non-learnable raw bytes of a non-f32
//! dtype, e.g. a `QuantLinear`'s packed U8 weight). Each wraps a fixed graph node
//! plus its host value; updates re-feed the node so the next eval sees them. All
//! cheap-clone (`Rc`), so a layer and the optimizer share one. Only `Param` is
//! reported by `parameters()` / touched by grad; the two buffers are persisted by
//! save/load so eval-mode state survives a round-trip.
use hodu_core::{Ctx, DType, Tensor};
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

/// Non-learnable, host-valued f32 Input leaf (e.g. BatchNorm running stats): like a
/// Param but excluded from parameters()/gradients -- a BUFFER the optimizer never
/// touches, updated host-side, and persisted by save/load. Cheap-clone (`Rc`).
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

/// Non-learnable, host-valued Input leaf holding RAW BYTES of a non-f32 dtype -- the
/// packed U8 quant weight in QuantLinear. Like Buffer but typed: fed as its true
/// Storage and persisted by save/load as raw bytes + a dtype tag, so a quantized weight
/// round-trips at its real size (not f32-expanded). Cheap-clone (`Rc`).
#[derive(Clone)]
pub struct QBuffer(Rc<QBufferInner>);

struct QBufferInner {
    ctx: Ctx,
    tensor: Tensor,
    shape: Vec<usize>,
    bytes: RefCell<Vec<u8>>,
}

impl QBuffer {
    /// A U8 Input leaf seeded with `bytes` (a packed quant weight of `shape`).
    pub fn u8(ctx: &Ctx, bytes: Vec<u8>, shape: Vec<usize>) -> QBuffer {
        let tensor = ctx.input_dtype(shape.clone(), DType::U8);
        ctx.feed_u8(tensor.node(), bytes.clone(), shape.clone());
        QBuffer(Rc::new(QBufferInner { ctx: ctx.clone(), tensor, shape, bytes: RefCell::new(bytes) }))
    }

    /// The graph handle for this buffer (use it in a forward pass).
    pub fn tensor(&self) -> &Tensor {
        &self.0.tensor
    }

    /// This buffer's shape.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// A copy of the current raw bytes.
    pub fn bytes(&self) -> Vec<u8> {
        self.0.bytes.borrow().clone()
    }

    /// Overwrite the raw bytes and re-feed them (host-side update / `load`).
    pub fn set_bytes(&self, bytes: Vec<u8>) {
        *self.0.bytes.borrow_mut() = bytes;
        self.0.ctx.feed_u8(self.0.tensor.node(), self.0.bytes.borrow().clone(), self.0.shape.clone());
    }
}
