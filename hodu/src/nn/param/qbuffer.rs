//! Non-learnable, host-valued Input leaf holding RAW BYTES of a non-f32 dtype -- the
//! packed U8 quant weight in QuantLinear. Like Buffer but typed: fed as its true
//! Storage and persisted by save/load as raw bytes + a dtype tag, so a quantized weight
//! round-trips at its real size (not f32-expanded). Cheap-clone (`Rc`).
use std::cell::RefCell;
use std::rc::Rc;

use hodu_core::{Ctx, DType, Tensor};

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
