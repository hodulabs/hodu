//! Optimizers. Grads come from `loss.grad(&param_tensors)` (built once); each step
//! re-realizes them with the current feeds (`grad_values`), then updates the host
//! values. Params are `Rc`-shared, so an update is visible to the next forward.
//! State (Adam's moments) is indexed by param position -- pass the params to the
//! optimizer in the same order used to build the grads.

mod adam;
mod grad;
mod scheduler;
mod sgd;

pub use adam::{Adam, AdamW};
pub use grad::{accumulate_grads, clip_grad_norm, grad_values, scale_grads};
pub use scheduler::{CosineAnnealingLR, LambdaLR, StepLR};
pub use sgd::Sgd;

use hodu_core::Error;

/// A named snapshot of an optimizer's mutable state (moments + step), keyed by name
/// so it round-trips through the `.hodu` container's `optim` tensors independent of
/// order. Used by `save_checkpoint`/`load_checkpoint` to resume a run with moments
/// and step intact. Names are stable for a fixed param count (the resume contract).
pub trait OptState {
    fn state_dict(&self) -> Vec<(String, Vec<f32>)>;
    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error>;
}

fn opt_err(msg: String) -> Error {
    Error::Shape { op: "optim load_state_dict", msg }
}

// Look up a named optimizer slot and validate its length.
pub(super) fn take_slot(sd: &[(String, Vec<f32>)], name: &str, len: usize) -> Result<Vec<f32>, Error> {
    let v = sd
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v)
        .ok_or_else(|| opt_err(format!("missing optim tensor '{name}'")))?;
    if v.len() != len {
        return Err(opt_err(format!("optim tensor '{name}' len {} != expected {len}", v.len())));
    }
    Ok(v.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Param;
    use hodu_core::Ctx;

    #[test]
    fn adam_first_step_is_lr_signed() {
        // First bias-corrected Adam step ~= -lr * sign(grad), independent of |grad|.
        let ctx = Ctx::cpu();
        let p = Param::new(&ctx, vec![0.0], vec![1]);
        let mut opt = Adam::new(vec![p.clone()], 0.1);
        opt.step(&[vec![2.0]]);
        assert!((p.value()[0] + 0.1).abs() < 1e-4, "got {}", p.value()[0]);
    }

    #[test]
    fn adamw_decays_weight() {
        // With zero grad, AdamW still shrinks the weight by lr*wd*w.
        let ctx = Ctx::cpu();
        let p = Param::new(&ctx, vec![1.0], vec![1]);
        let mut opt = AdamW::new(vec![p.clone()], 0.1, 0.5);
        opt.step(&[vec![0.0]]);
        // delta = lr*wd*w = 0.1*0.5*1.0 = 0.05 -> value 0.95.
        assert!((p.value()[0] - 0.95).abs() < 1e-5, "got {}", p.value()[0]);
    }

    #[test]
    fn scheduler_drives_optimizer_lr() {
        // StepLR (decay every epoch) fed into opt.set_lr moves the real optimizer's lr.
        let ctx = Ctx::cpu();
        let p = Param::new(&ctx, vec![0.0], vec![1]);
        let opt = Sgd::new(vec![p], 1.0);
        let mut sched = StepLR::new(1.0, 1, 0.5);
        assert!((opt.lr() - 1.0).abs() < 1e-9);
        opt.set_lr(sched.step()); // epoch 1 -> 0.5
        assert!((opt.lr() - 0.5).abs() < 1e-9, "lr {}", opt.lr());
        opt.set_lr(sched.step()); // epoch 2 -> 0.25
        assert!((opt.lr() - 0.25).abs() < 1e-9, "lr {}", opt.lr());
    }

    #[test]
    fn sgd_momentum_accumulates() {
        // Constant grad g=1, mu=0.9, lr=0.1: step1 v=1 -> -0.1; step2 v=1.9 -> -0.19.
        let ctx = Ctx::cpu();
        let p = Param::new(&ctx, vec![0.0], vec![1]);
        let opt = Sgd::with_momentum(vec![p.clone()], 0.1, 0.9, 0.0);
        opt.step(&[vec![1.0]]);
        assert!((p.value()[0] + 0.1).abs() < 1e-6, "got {}", p.value()[0]);
        opt.step(&[vec![1.0]]);
        assert!((p.value()[0] + 0.29).abs() < 1e-6, "got {}", p.value()[0]); // -0.1 -0.19
    }
}
