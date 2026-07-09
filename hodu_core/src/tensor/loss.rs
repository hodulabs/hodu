//! Loss ops on `Tensor` (raw engine ops; `hodu::loss` wraps them as free fns).
use kurumi::Error;

use crate::Tensor;

impl Tensor {
    /// Softmax cross-entropy over `axis`: `-sum(targets * log_softmax(self))`,
    /// numerically stable. `targets` are class probabilities (one-hot for hard
    /// labels), same shape as `self`. Returns the per-example loss (axis reduced).
    pub fn cross_entropy(&self, targets: &Tensor, axis: usize) -> Result<Tensor, Error> {
        let (ln, tn) = (self.node(), targets.node());
        self.ctx().build(|g| g.cross_entropy(ln, tn, axis))
    }
}
