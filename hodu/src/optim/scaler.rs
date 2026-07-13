//! Dynamic loss scaling (GradScaler), host-side. Scale the loss up so small grads
//! survive low-precision math, then divide the host grads back down and skip the step
//! whenever any grad is non-finite (an overflow), backing the scale off; grow it again
//! after a run of clean steps.
use hodu_core::Tensor;

/// Host-side dynamic loss scaler. `scale(loss)` multiplies the loss so backward yields
/// scaled grads; after backward, `step` divides the host grads by `scale`, checks them
/// all finite, and either applies the optimizer step (growing the scale after
/// `growth_interval` good steps) or SKIPS it and multiplies the scale by `backoff_factor`.
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    good: usize, // consecutive finite steps since the last growth/backoff
}

impl GradScaler {
    /// Defaults: `growth_factor = 2.0`, `backoff_factor = 0.5`, `growth_interval = 2000`.
    pub fn new(init_scale: f32) -> GradScaler {
        GradScaler { scale: init_scale, growth_factor: 2.0, backoff_factor: 0.5, growth_interval: 2000, good: 0 }
    }

    pub fn scale_factor(&self) -> f32 {
        self.scale
    }

    /// Scale a loss up by the current factor (call before backward).
    pub fn scale(&self, loss: &Tensor) -> Tensor {
        loss * self.scale
    }

    /// Divide `grads` by the current scale, then: if all finite, run `apply` (the
    /// optimizer's step) and grow the scale after `growth_interval` clean steps, returning
    /// `true`; if any grad is non-finite, SKIP `apply`, back the scale off, and return `false`.
    pub fn step(&mut self, grads: &mut [Vec<f32>], apply: impl FnOnce(&[Vec<f32>])) -> bool {
        let inv = 1.0 / self.scale;
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= inv;
            }
        }
        let finite = grads.iter().all(|g| g.iter().all(|x| x.is_finite()));
        if !finite {
            self.scale *= self.backoff_factor;
            self.good = 0;
            return false;
        }
        apply(grads);
        self.good += 1;
        if self.good >= self.growth_interval {
            self.scale *= self.growth_factor;
            self.good = 0;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Param;
    use crate::optim::{Sgd, grad_values};
    use hodu_core::Ctx;

    #[test]
    fn inf_grad_skips_and_backs_off_then_finite_applies() {
        let ctx = Ctx::cpu();
        let p = Param::new(&ctx, vec![1.0], vec![1]);
        let x = ctx.input(vec![1]);
        let loss = p.tensor().try_mul(&x).unwrap(); // grad wrt p = x
        let grads = loss.grad(&[p.tensor()]).unwrap();
        let opt = Sgd::new(vec![p.clone()], 0.1);
        let mut scaler = GradScaler::new(1.0);

        // scale() multiplies the loss: p=1, x=3 -> loss 3, factor 1 -> 3.
        ctx.feed(x.node(), vec![3.0], vec![1]);
        assert!((scaler.scale(&loss).item() - 3.0).abs() < 1e-6);

        // x = inf -> grad inf -> step skipped, scale halved, param frozen.
        ctx.feed(x.node(), vec![f32::INFINITY], vec![1]);
        let before = p.value();
        let mut g = grad_values(&grads);
        assert!(!scaler.step(&mut g, |gg| opt.step(gg)), "inf grad must skip");
        assert_eq!(p.value(), before, "skipped step must not move the param");
        assert!((scaler.scale_factor() - 0.5).abs() < 1e-9, "scale must back off to 0.5");

        // x = 2 -> finite grad -> step applied, param moves.
        ctx.feed(x.node(), vec![2.0], vec![1]);
        let mut g = grad_values(&grads);
        assert!(scaler.step(&mut g, |gg| opt.step(gg)), "finite grad must apply");
        assert!(p.value() != before, "applied step must move the param");
    }
}
