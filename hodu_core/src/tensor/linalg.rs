//! Linear-algebra ops on `Tensor`. Just matmul for now; inv/solve/det/svd wraps
//! land here as the engine surfaces them.
use kurumi::{DType, Error};

use crate::Tensor;

impl Tensor {
    /// 2-D / rank-N@2-D matmul: contract this tensor's last axis with `rhs`'s
    /// second-to-last (`dot_general`, no batch dims).
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        let (lc, rc) = (self.rank().saturating_sub(1), rhs.rank().saturating_sub(2));
        let (a, b) = (self.node(), rhs.node());
        self.ctx().build(|g| g.dot_general(a, b, vec![lc], vec![rc], vec![], vec![]))
    }

    /// Weight-only quantized matmul: `self[M,K] x dequant(qweight)[N,K]^T -> [M,N]`.
    /// `qweight` is a U8-packed weight, `scales`/`mins` its per-group scale (and, when
    /// asymmetric, min); `mins = None` is symmetric. `scales`/`mins` are cast to F16
    /// here if not already (the op requires F16). The quantized weight is frozen -- no
    /// gradient flows through it.
    pub fn quant_matmul(
        &self,
        qweight: &Tensor,
        scales: &Tensor,
        mins: Option<&Tensor>,
        bits: u8,
        group_size: usize,
    ) -> Result<Tensor, Error> {
        let f16 = |t: &Tensor| {
            if t.dtype() == DType::F16 { t.clone() } else { t.cast(DType::F16) }
        };
        let (a, qw) = (self.node(), qweight.node());
        let sc = f16(scales).node();
        let mn = mins.map(|m| f16(m).node());
        self.ctx().build(|g| g.quant_matmul(a, qw, sc, mn, bits, group_size))
    }
}
