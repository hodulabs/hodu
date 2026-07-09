//! SGD with optional momentum + coupled weight decay.
use std::cell::{Cell, RefCell};

use hodu_core::Error;

use crate::nn::Param;

use super::{OptState, opt_err, take_slot};

/// SGD, optionally with momentum + weight decay: `g' = g + wd*p`, `v = mu*v + g'`,
/// `p -= lr*v` (plain SGD when `mu = wd = 0`). `lr` is a `Cell` so an LR scheduler
/// can retune it via [`Sgd::set_lr`] without `&mut`.
pub struct Sgd {
    params: Vec<Param>,
    lr: Cell<f32>,
    momentum: f32,
    wd: f32,
    vel: RefCell<Vec<Vec<f32>>>,
}

impl Sgd {
    pub fn new(params: Vec<Param>, lr: f32) -> Sgd {
        Sgd::with_momentum(params, lr, 0.0, 0.0)
    }

    /// SGD with `momentum` and (coupled) `weight_decay`.
    pub fn with_momentum(params: Vec<Param>, lr: f32, momentum: f32, weight_decay: f32) -> Sgd {
        let vel = params.iter().map(|p| vec![0.0; p.value().len()]).collect();
        Sgd { params, lr: Cell::new(lr), momentum, wd: weight_decay, vel: RefCell::new(vel) }
    }

    pub fn lr(&self) -> f32 {
        self.lr.get()
    }
    pub fn set_lr(&self, lr: f32) {
        self.lr.set(lr);
    }

    /// Apply one step from grad values aligned with `self.params`.
    pub fn step(&self, grads: &[Vec<f32>]) {
        let (lr, mu, wd) = (self.lr.get(), self.momentum, self.wd);
        if mu == 0.0 && wd == 0.0 {
            for (p, g) in self.params.iter().zip(grads) {
                p.apply_grad(lr, g);
            }
            return;
        }
        let mut vel = self.vel.borrow_mut();
        for (i, (p, g)) in self.params.iter().zip(grads).enumerate() {
            let pv = if wd != 0.0 { p.value() } else { Vec::new() };
            let v = &mut vel[i];
            let mut delta = vec![0.0f32; g.len()];
            for j in 0..g.len() {
                let gj = g[j] + if wd != 0.0 { wd * pv[j] } else { 0.0 };
                v[j] = mu * v[j] + gj;
                delta[j] = lr * v[j];
            }
            p.apply_grad(1.0, &delta);
        }
    }
}

impl OptState for Sgd {
    fn state_dict(&self) -> Vec<(String, Vec<f32>)> {
        // Leading count sentinel so even a zero-param Sgd writes one always-read row:
        // `load` on a file with no Sgd state then Errs (parity with Adam always reading
        // "step") instead of a silent no-op, and it catches a resume into a wrong-sized
        // model (fewer/more params than the checkpoint).
        let vel = self.vel.borrow();
        let mut out = vec![("vel.count".to_string(), vec![vel.len() as f32])];
        out.extend(vel.iter().enumerate().map(|(i, v)| (format!("vel.{i}"), v.clone())));
        out
    }
    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error> {
        let mut vel = self.vel.borrow_mut();
        let n = take_slot(sd, "vel.count", 1)?[0] as usize;
        if n != vel.len() {
            return Err(opt_err(format!("Sgd state has {n} velocity buffers but the model has {}", vel.len())));
        }
        for (i, slot) in vel.iter_mut().enumerate() {
            let len = slot.len();
            *slot = take_slot(sd, &format!("vel.{i}"), len)?;
        }
        Ok(())
    }
}
