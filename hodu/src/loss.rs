//! Loss functions returning a scalar (mean-reduced over the batch). Thin wraps of
//! kurumi's numerically-stable loss ops.
use hodu_core::{Error, Tensor};

/// Mean squared error `mean((pred - target)^2)`.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    pred.try_sub(target)?.square().mean_all()
}

/// Softmax cross-entropy over the last (class) axis, averaged over the batch.
/// `targets` are class probabilities (one-hot for hard labels), same shape as
/// `logits`.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor, Error> {
    let axis = logits.rank().saturating_sub(1);
    logits.cross_entropy(targets, axis)?.mean_all()
}

/// Mean absolute error `mean(|pred - target|)`.
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    pred.l1_loss(target)?.mean_all()
}

/// Huber loss (smooth L1): quadratic within `delta`, linear beyond, mean-reduced.
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor, Error> {
    pred.huber_loss(target, delta)?.mean_all()
}

/// Binary cross-entropy on probabilities `pred in (0,1)`, mean-reduced.
pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    pred.bce_loss(target)?.mean_all()
}

/// Binary cross-entropy from raw `logits` (numerically stable), mean-reduced.
pub fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    logits.bce_with_logits(target)?.mean_all()
}

/// Hinge loss `max(0, 1 - pred*target)` (`target in {-1,+1}`), mean-reduced.
pub fn hinge_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    pred.hinge_loss(target)?.mean_all()
}

/// KL divergence `sum(p * log(p/q))` (elementwise, mean-reduced).
pub fn kl_div(p: &Tensor, q: &Tensor) -> Result<Tensor, Error> {
    p.kl_div(q)?.mean_all()
}

/// Negative log-likelihood `mean(-sum(target * log_probs))` over the last (class)
/// axis. `log_probs` are log-probabilities (e.g. from `log_softmax`), `target`
/// one-hot.
pub fn nll_loss(log_probs: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    let axis = log_probs.rank().saturating_sub(1);
    log_probs.nll_loss(target, axis)?.mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hodu_core::Ctx;

    #[test]
    fn l1_and_huber_match_hand_values() {
        let ctx = Ctx::cpu();
        let pred = ctx.constant(vec![1.0, 2.0], vec![2]);
        let tgt = ctx.constant(vec![0.0, 0.0], vec![2]);
        // |1|+|2| /2 = 1.5
        assert!((l1_loss(&pred, &tgt).unwrap().item() - 1.5).abs() < 1e-6);
        // delta=1: 0.5*1^2=0.5 ; 1*(2-0.5)=1.5 ; mean=1.0
        assert!((huber_loss(&pred, &tgt, 1.0).unwrap().item() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bce_with_logits_at_zero_is_ln2() {
        // sigmoid(0)=0.5 -> per-element BCE = -ln(0.5) = ln2, any target.
        let ctx = Ctx::cpu();
        let logits = ctx.constant(vec![0.0, 0.0], vec![2]);
        let tgt = ctx.constant(vec![1.0, 0.0], vec![2]);
        let l = bce_with_logits(&logits, &tgt).unwrap().item();
        assert!((l - 2.0_f32.ln()).abs() < 1e-5, "got {l}");
    }

    #[test]
    fn hinge_is_hand_value() {
        // max(0, 1 - pred*target): [0.5*1, (-0.5)*(-1)] -> [0.5, 0.5], mean 0.5.
        let ctx = Ctx::cpu();
        let pred = ctx.constant(vec![0.5, -0.5], vec![2]);
        let tgt = ctx.constant(vec![1.0, -1.0], vec![2]);
        let l = hinge_loss(&pred, &tgt).unwrap().item();
        assert!((l - 0.5).abs() < 1e-6, "got {l}");
    }

    #[test]
    fn kl_div_zero_when_equal_and_finite() {
        // p==q -> sum(p*log(p/q)) = 0; and a general case stays finite.
        let ctx = Ctx::cpu();
        let p = ctx.constant(vec![0.5, 0.5], vec![2]);
        assert!(kl_div(&p, &p).unwrap().item().abs() < 1e-6);
        let q = ctx.constant(vec![0.3, 0.7], vec![2]);
        assert!(kl_div(&p, &q).unwrap().item().is_finite());
    }

    #[test]
    fn bce_and_nll_are_finite() {
        let ctx = Ctx::cpu();
        let pr = ctx.constant(vec![0.9, 0.1], vec![2]);
        let tg = ctx.constant(vec![1.0, 0.0], vec![2]);
        assert!(bce_loss(&pr, &tg).unwrap().item().is_finite());
        let logits = ctx.constant(vec![2.0, -1.0, 0.0, 3.0], vec![2, 2]);
        let onehot = ctx.constant(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let lp = logits.log_softmax(1).unwrap();
        assert!(nll_loss(&lp, &onehot).unwrap().item() > 0.0);
    }
}
