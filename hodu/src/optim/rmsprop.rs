//! RMSprop: per-parameter squared-grad EMA, step scaled by its root.
use crate::nn::Param;
use crate::optim::{OptState, opt_err, take_slot};
use hodu_core::Error;
use std::cell::{Cell, RefCell};

/// RMSprop: keep an EMA of squared grads `sq = alpha*sq + (1-alpha)*g^2`, then
/// `p -= lr * g / (sqrt(sq) + eps)`. `lr` is a `Cell` so an LR scheduler can retune it
/// via [`RMSprop::set_lr`] without `&mut` (parity with [`crate::optim::Sgd`]).
pub struct RMSprop {
    params: Vec<Param>,
    lr: Cell<f32>,
    alpha: f32,
    eps: f32,
    sq: RefCell<Vec<Vec<f32>>>,
}

impl RMSprop {
    /// Defaults: `alpha = 0.99`, `eps = 1e-8`.
    pub fn new(params: Vec<Param>, lr: f32) -> RMSprop {
        RMSprop::with_params(params, lr, 0.99, 1e-8)
    }

    /// RMSprop with an explicit `alpha` (squared-grad EMA decay) and `eps`.
    pub fn with_params(params: Vec<Param>, lr: f32, alpha: f32, eps: f32) -> RMSprop {
        let sq = params.iter().map(|p| vec![0.0; p.value().len()]).collect();
        RMSprop { params, lr: Cell::new(lr), alpha, eps, sq: RefCell::new(sq) }
    }

    pub fn lr(&self) -> f32 {
        self.lr.get()
    }
    pub fn set_lr(&self, lr: f32) {
        self.lr.set(lr);
    }

    /// Apply one step from grad values aligned with `self.params`.
    pub fn step(&self, grads: &[Vec<f32>]) {
        let (lr, alpha, eps) = (self.lr.get(), self.alpha, self.eps);
        let mut sq = self.sq.borrow_mut();
        for (i, (p, g)) in self.params.iter().zip(grads).enumerate() {
            let s = &mut sq[i];
            let mut delta = vec![0.0f32; g.len()];
            for j in 0..g.len() {
                s[j] = alpha * s[j] + (1.0 - alpha) * g[j] * g[j];
                delta[j] = lr * g[j] / (s[j].sqrt() + eps);
            }
            p.apply_grad(1.0, &delta); // delta already scaled by lr
        }
    }
}

impl OptState for RMSprop {
    fn state_dict(&self) -> Vec<(String, Vec<f32>)> {
        // Leading count sentinel (like Sgd's `vel.count`) so a load on a file with no
        // RMSprop state Errs instead of silently no-op'ing, and catches a wrong-sized model.
        let sq = self.sq.borrow();
        let mut out = vec![("sq.count".to_string(), vec![sq.len() as f32])];
        out.extend(sq.iter().enumerate().map(|(i, s)| (format!("sq.{i}"), s.clone())));
        out
    }
    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error> {
        let mut sq = self.sq.borrow_mut();
        let n = take_slot(sd, "sq.count", 1)?[0] as usize;
        if n != sq.len() {
            return Err(opt_err(format!("RMSprop state has {n} sq buffers but the model has {}", sq.len())));
        }
        for (i, slot) in sq.iter_mut().enumerate() {
            let len = slot.len();
            *slot = take_slot(sd, &format!("sq.{i}"), len)?;
        }
        Ok(())
    }
}
