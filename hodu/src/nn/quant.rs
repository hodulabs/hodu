//! Weight-only quantized linear: `y = x @ dequant(qweight)^T + b`. The deploy path --
//! smaller persisted weights on the same static graph as f32 `Linear`. The weight is
//! quantized once (kurumi group-wise int8/int4 affine) into a frozen U8 packed tensor
//! plus per-group scales (and, asymmetric, mins); forward runs the fused `quant_matmul`
//! op. Only the bias is learnable -- the quantized weight is non-differentiable (frozen),
//! held as non-learnable buffers so it never gets a gradient but still round-trips.
//!
//! The packed weight is a true U8 `QBuffer` (persisted as raw bytes -> the size win).
//! scales/mins are stored as f32 `Buffer`s and cast to F16 in-graph (the widened f16 is
//! recovered exactly by the f32->f16 cast); this keeps the frontend `half`-free while the
//! op still sees the F16 it requires.
use crate::nn::linear::Linear;
use crate::nn::{Buffer, Module, Param, QBuffer, QuantDescriptor};
use hodu_core::kurumi::quantize;
use hodu_core::{Ctx, Error, Tensor};

/// A quantized affine layer. Build with [`QuantLinear::from_linear`]; `forward` matches
/// an f32 `Linear` within quantization error.
pub struct QuantLinear {
    qweight: QBuffer,     // U8 packed, [out, in * bits / 8]
    scales: Buffer,       // f32 (widened f16), [out, in / group_size]
    mins: Option<Buffer>, // f32, [out, in / group_size]; Some = asymmetric
    bias: Param,          // f32, [out]
    bits: u8,
    group_size: usize,
}

impl QuantLinear {
    /// Quantize an f32 [`Linear`]'s weight to `bits` (8/4) with `group_size` columns per
    /// scale; `symmetric` picks the signed (no min) vs asymmetric (scale+min) scheme.
    /// The Linear's `in` must be a multiple of `group_size` (the quant group runs along
    /// the contraction axis). The bias is carried over as the sole learnable param.
    pub fn from_linear(lin: &Linear, bits: u8, group_size: usize, symmetric: bool) -> Result<QuantLinear, Error> {
        let w = lin.weight();
        let (in_f, out) = (w.shape()[0], w.shape()[1]);
        if group_size == 0 || !in_f.is_multiple_of(group_size) {
            return Err(Error::Shape {
                op: "QuantLinear::from_linear",
                msg: format!("in ({in_f}) must be a nonzero multiple of group_size ({group_size})"),
            });
        }
        // Linear stores w row-major as [in, out]; quantize wants the [out, in] weight
        // (rows = out = N, cols = in = K, groups along K), so transpose first.
        let wv = w.value();
        let mut wt = vec![0f32; out * in_f];
        for i in 0..in_f {
            for j in 0..out {
                wt[j * in_f + i] = wv[i * out + j];
            }
        }
        let q = quantize(&wt, out, in_f, bits, group_size, symmetric);

        let ctx: Ctx = w.tensor().ctx().clone();
        let ng = in_f / group_size;
        let packed_cols = in_f * bits as usize / 8;
        let qweight = QBuffer::u8(&ctx, q.packed, vec![out, packed_cols]);
        let scales = Buffer::new(&ctx, q.scales.iter().map(|s| s.to_f32()).collect(), vec![out, ng]);
        let mins = q.mins.map(|m| Buffer::new(&ctx, m.iter().map(|v| v.to_f32()).collect(), vec![out, ng]));
        let bias = Param::new(&ctx, lin.bias().map(|b| b.value()).unwrap_or_else(|| vec![0.0; out]), vec![out]);
        Ok(QuantLinear { qweight, scales, mins, bias, bits, group_size })
    }
}

impl Module for QuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let mins = self.mins.as_ref().map(|m| m.tensor());
        let y = x.quant_matmul(self.qweight.tensor(), self.scales.tensor(), mins, self.bits, self.group_size)?;
        y.try_add(self.bias.tensor()) // [M, out] + [out] broadcasts
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.bias.clone()] // the quantized weight is frozen -> only the bias trains
    }
    fn buffers(&self) -> Vec<Buffer> {
        let mut v = vec![self.scales.clone()];
        v.extend(self.mins.clone());
        v
    }
    fn byte_buffers(&self) -> Vec<QBuffer> {
        vec![self.qweight.clone()]
    }
    fn quant_descriptors(&self, prefix: &str) -> Vec<QuantDescriptor> {
        // FQNs must match the `number()` counter the named_* walks use: one running index over
        // params (bias -> 0), then buffers (scales, then mins if asymmetric), then byte-buffers
        // (qweight). So scales = prefix+np, mins = prefix+(np+1), qweight = prefix+(np+nb).
        let np = self.parameters().len(); // bias -> 1
        let nb = self.buffers().len(); // scales [+ mins] -> 1 or 2
        vec![QuantDescriptor {
            weight_fqn: format!("{prefix}{}", np + nb),
            bits: self.bits,
            group_size: self.group_size,
            symmetric: self.mins.is_none(),
            scales_fqn: format!("{prefix}{np}"),
            mins_fqn: self.mins.as_ref().map(|_| format!("{prefix}{}", np + 1)),
        }]
    }
}
