//! Loss ops on `Tensor` (raw engine ops; `hodu::loss` wraps them as free fns).
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    /// Softmax cross-entropy over `axis`: `-sum(targets * log_softmax(self))`,
    /// numerically stable. `targets` are class probabilities (one-hot for hard
    /// labels), same shape as `self`. Returns the per-example loss (axis reduced).
    pub fn cross_entropy(&self, targets: &Tensor, axis: usize) -> Result<Tensor, Error> {
        let (ln, tn) = (self.node(), targets.node());
        self.ctx().build(|g| g.cross_entropy(ln, tn, axis))
    }

    /// Absolute error `|self - target|`, elementwise (unreduced; `hodu::loss::l1_loss` means it).
    pub fn l1_loss(&self, target: &Tensor) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.l1_loss(a, b))
    }

    /// Huber (smooth-L1) loss: quadratic within `delta`, linear beyond, elementwise (unreduced).
    pub fn huber_loss(&self, target: &Tensor, delta: f32) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.huber_loss(a, b, delta))
    }

    /// Binary cross-entropy on probabilities `self in (0,1)`, elementwise (unreduced).
    pub fn bce_loss(&self, target: &Tensor) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.bce_loss(a, b))
    }

    /// Binary cross-entropy from raw logits `self` (numerically stable), elementwise (unreduced).
    pub fn bce_with_logits(&self, target: &Tensor) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.bce_with_logits(a, b))
    }

    /// Hinge loss `max(0, 1 - self*target)` (`target in {-1,+1}`), elementwise (unreduced).
    pub fn hinge_loss(&self, target: &Tensor) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.hinge_loss(a, b))
    }

    /// KL divergence `self * log(self/other)`, elementwise (unreduced).
    pub fn kl_div(&self, other: &Tensor) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), other.node());
        self.ctx().build(|g| g.kl_div(a, b))
    }

    /// Negative log-likelihood over `axis`: `-sum(target * self)` for log-prob `self` (axis reduced).
    pub fn nll_loss(&self, target: &Tensor, axis: usize) -> Result<Tensor, Error> {
        let (a, b) = (self.node(), target.node());
        self.ctx().build(|g| g.nll_loss(a, b, axis))
    }
}
