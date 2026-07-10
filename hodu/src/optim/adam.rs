//! Adam and AdamW (decoupled weight decay), over shared `AdamState` machinery.
use crate::nn::Param;
use crate::optim::{OptState, take_slot};
use hodu_core::Error;

/// Adam: per-parameter first/second moment estimates with bias correction. For
/// weight decay use [`AdamW`] (decoupled -- the modern default); plain Adam here
/// has none.
pub struct Adam {
    st: AdamState,
}

impl Adam {
    /// Defaults: betas `(0.9, 0.999)`, eps `1e-8`.
    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        Adam { st: AdamState::new(params, lr, 0.0) }
    }
    pub fn step(&mut self, grads: &[Vec<f32>]) {
        self.st.step(grads, false);
    }
    pub fn lr(&self) -> f32 {
        self.st.lr
    }
    pub fn set_lr(&mut self, lr: f32) {
        self.st.lr = lr;
    }
}

/// AdamW: Adam with decoupled weight decay (`p -= lr * weight_decay * p`).
pub struct AdamW {
    st: AdamState,
}

impl AdamW {
    pub fn new(params: Vec<Param>, lr: f32, weight_decay: f32) -> AdamW {
        AdamW { st: AdamState::new(params, lr, weight_decay) }
    }
    pub fn step(&mut self, grads: &[Vec<f32>]) {
        self.st.step(grads, true);
    }
    pub fn lr(&self) -> f32 {
        self.st.lr
    }
    pub fn set_lr(&mut self, lr: f32) {
        self.st.lr = lr;
    }
}

impl OptState for Adam {
    fn state_dict(&self) -> Vec<(String, Vec<f32>)> {
        self.st.state_dict()
    }
    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error> {
        self.st.load_state_dict(sd)
    }
}

impl OptState for AdamW {
    fn state_dict(&self) -> Vec<(String, Vec<f32>)> {
        self.st.state_dict()
    }
    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error> {
        self.st.load_state_dict(sd)
    }
}

// Shared Adam machinery. `m`/`v` are per-param moment buffers indexed by position.
struct AdamState {
    params: Vec<Param>,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamState {
    fn new(params: Vec<Param>, lr: f32, wd: f32) -> AdamState {
        let (m, v) = params
            .iter()
            .map(|p| {
                let z = vec![0.0f32; p.value().len()];
                (z.clone(), z)
            })
            .unzip();
        AdamState { params, lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd, t: 0, m, v }
    }

    fn step(&mut self, grads: &[Vec<f32>], decoupled: bool) {
        self.t += 1;
        let (b1, b2, eps, lr, wd) = (self.b1, self.b2, self.eps, self.lr, self.wd);
        let bc1 = 1.0 - b1.powi(self.t as i32); // bias corrections
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, (p, g)) in self.params.iter().zip(grads).enumerate() {
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let val = if decoupled { p.value() } else { Vec::new() };
            let mut delta = vec![0.0f32; g.len()];
            for j in 0..g.len() {
                m[j] = b1 * m[j] + (1.0 - b1) * g[j];
                v[j] = b2 * v[j] + (1.0 - b2) * g[j] * g[j];
                let mhat = m[j] / bc1;
                let vhat = v[j] / bc2;
                let mut d = lr * mhat / (vhat.sqrt() + eps);
                if decoupled {
                    d += lr * wd * val[j];
                }
                delta[j] = d;
            }
            p.apply_grad(1.0, &delta); // delta already scaled by lr
        }
    }

    fn state_dict(&self) -> Vec<(String, Vec<f32>)> {
        // step as f32 is exact below 2^24 steps; widen to a typed u64 tensor if longer runs need it.
        let mut out = vec![("step".to_string(), vec![self.t as f32])];
        for (i, m) in self.m.iter().enumerate() {
            out.push((format!("m.{i}"), m.clone()));
        }
        for (i, v) in self.v.iter().enumerate() {
            out.push((format!("v.{i}"), v.clone()));
        }
        out
    }

    fn load_state_dict(&mut self, sd: &[(String, Vec<f32>)]) -> Result<(), Error> {
        self.t = take_slot(sd, "step", 1)?[0] as u64;
        for (i, m) in self.m.iter_mut().enumerate() {
            let len = m.len();
            *m = take_slot(sd, &format!("m.{i}"), len)?;
        }
        for (i, v) in self.v.iter_mut().enumerate() {
            let len = v.len();
            *v = take_slot(sd, &format!("v.{i}"), len)?;
        }
        Ok(())
    }
}
