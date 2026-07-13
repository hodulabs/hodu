//! LR schedulers. Each is a pure LR curve over an epoch counter: call `step()` once
//! per epoch and hand the result to `opt.set_lr(..)` -- no borrow of the optimizer,
//! which the static `Cell`/`&mut` lr setters make unnecessary.

/// A scheduler's resumable epoch counter, so `save_checkpoint`/`load_checkpoint` can
/// persist and restore where a run's LR curve was (the `sched.last_epoch` row).
pub trait SchedState {
    fn last_epoch(&self) -> usize;
    fn set_last_epoch(&mut self, epoch: usize);
}

/// Decay LR by `gamma` every `step_size` epochs: `base * gamma^(epoch/step_size)`.
pub struct StepLR {
    base: f32,
    step_size: usize,
    gamma: f32,
    epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> StepLR {
        StepLR { base: base_lr, step_size, gamma, epoch: 0 }
    }
    pub fn lr(&self) -> f32 {
        self.base * self.gamma.powi((self.epoch / self.step_size) as i32)
    }
    /// Advance one epoch; returns the new LR.
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.lr()
    }
}

/// Decay LR by `gamma` at each of a sorted list of `milestones`:
/// `base * gamma^(count of milestones <= epoch)`.
pub struct MultiStepLR {
    base: f32,
    milestones: Vec<usize>,
    gamma: f32,
    epoch: usize,
}

impl MultiStepLR {
    /// `milestones` is sorted on construction, so any input order works.
    pub fn new(base_lr: f32, milestones: Vec<usize>, gamma: f32) -> MultiStepLR {
        let mut milestones = milestones;
        milestones.sort_unstable();
        MultiStepLR { base: base_lr, milestones, gamma, epoch: 0 }
    }
    pub fn lr(&self) -> f32 {
        let hit = self.milestones.iter().filter(|&&m| m <= self.epoch).count();
        self.base * self.gamma.powi(hit as i32)
    }
    /// Advance one epoch; returns the new LR.
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.lr()
    }
}

/// Cosine anneal from `base` to `eta_min` over `t_max` epochs:
/// `eta_min + (base - eta_min) * 0.5*(1 + cos(pi*t/t_max))`.
pub struct CosineAnnealingLR {
    base: f32,
    t_max: usize,
    eta_min: f32,
    epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> CosineAnnealingLR {
        CosineAnnealingLR { base: base_lr, t_max, eta_min, epoch: 0 }
    }
    pub fn lr(&self) -> f32 {
        let t = (self.epoch.min(self.t_max) as f32) / self.t_max as f32;
        self.eta_min + (self.base - self.eta_min) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
    }
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.lr()
    }
}

/// LR = `base * f(epoch)` for a caller-supplied factor function.
pub struct LambdaLR {
    base: f32,
    f: Box<dyn Fn(usize) -> f32>,
    epoch: usize,
}

impl LambdaLR {
    pub fn new(base_lr: f32, f: impl Fn(usize) -> f32 + 'static) -> LambdaLR {
        LambdaLR { base: base_lr, f: Box::new(f), epoch: 0 }
    }
    pub fn lr(&self) -> f32 {
        self.base * (self.f)(self.epoch)
    }
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.lr()
    }
}

// last_epoch is each scheduler's `epoch` field -- the only resumable state.
macro_rules! sched_state {
    ($t:ty) => {
        impl SchedState for $t {
            fn last_epoch(&self) -> usize {
                self.epoch
            }
            fn set_last_epoch(&mut self, epoch: usize) {
                self.epoch = epoch;
            }
        }
    };
}
sched_state!(StepLR);
sched_state!(MultiStepLR);
sched_state!(CosineAnnealingLR);
sched_state!(LambdaLR);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steplr_curve() {
        let mut s = StepLR::new(1.0, 2, 0.5);
        assert!((s.lr() - 1.0).abs() < 1e-9); // epoch 0
        s.step(); // 1
        assert!((s.lr() - 1.0).abs() < 1e-9);
        s.step(); // 2 -> decays once
        assert!((s.lr() - 0.5).abs() < 1e-9);
        s.step();
        s.step(); // 4 -> twice
        assert!((s.lr() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn multistep_decays_at_each_milestone() {
        // milestones fed out of order; decay by 0.1 at epochs 2 and 5.
        let mut s = MultiStepLR::new(1.0, vec![5, 2], 0.1);
        assert!((s.lr() - 1.0).abs() < 1e-9); // epoch 0
        s.step(); // 1
        assert!((s.lr() - 1.0).abs() < 1e-9);
        s.step(); // 2 -> once
        assert!((s.lr() - 0.1).abs() < 1e-7, "lr {}", s.lr());
        for _ in 0..3 {
            s.step(); // 5 -> twice
        }
        assert!((s.lr() - 0.01).abs() < 1e-7, "lr {}", s.lr());
    }

    #[test]
    fn cosine_starts_at_base_hits_eta_min_at_tmax() {
        let mut s = CosineAnnealingLR::new(1.0, 4, 0.1);
        assert!((s.lr() - 1.0).abs() < 1e-6); // epoch 0: eta_min + (base-eta_min)*1 = base
        for _ in 0..4 {
            s.step();
        }
        assert!((s.lr() - 0.1).abs() < 1e-6, "lr at t_max {}", s.lr()); // anneals to eta_min
    }

    #[test]
    fn sched_state_roundtrips_epoch() {
        // last_epoch/set_last_epoch expose the resumable counter for checkpointing.
        let mut s = StepLR::new(1.0, 1, 0.5);
        s.step();
        s.step();
        assert_eq!(s.last_epoch(), 2);
        let mut r = StepLR::new(1.0, 1, 0.5);
        r.set_last_epoch(s.last_epoch());
        assert!((r.lr() - s.lr()).abs() < 1e-9); // restored curve position
    }

    #[test]
    fn lambdalr_applies_fn() {
        // base 2.0, f(e)=0.5^e -> lr(0)=2, after one step lr=2*0.5=1.
        let mut s = LambdaLR::new(2.0, |e| 0.5_f32.powi(e as i32));
        assert!((s.lr() - 2.0).abs() < 1e-6);
        assert!((s.step() - 1.0).abs() < 1e-6);
    }
}
