//! LR schedulers. Each is a pure LR curve over an epoch counter: call `step()` once
//! per epoch and hand the result to `opt.set_lr(..)` -- no borrow of the optimizer,
//! which the static `Cell`/`&mut` lr setters make unnecessary.

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

/// Cosine anneal from `base` to 0 over `t_max` epochs: `base * 0.5*(1+cos(pi*t/t_max))`.
pub struct CosineAnnealingLR {
    base: f32,
    t_max: usize,
    epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize) -> CosineAnnealingLR {
        CosineAnnealingLR { base: base_lr, t_max, epoch: 0 }
    }
    pub fn lr(&self) -> f32 {
        let t = (self.epoch.min(self.t_max) as f32) / self.t_max as f32;
        self.base * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
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
    fn cosine_starts_at_base_hits_zero_at_tmax() {
        let mut s = CosineAnnealingLR::new(1.0, 4);
        assert!((s.lr() - 1.0).abs() < 1e-6); // epoch 0: 0.5*(1+cos0)=1
        for _ in 0..4 {
            s.step();
        }
        assert!(s.lr().abs() < 1e-6, "lr at t_max {}", s.lr()); // 0.5*(1+cos pi)=0
    }

    #[test]
    fn lambdalr_applies_fn() {
        // base 2.0, f(e)=0.5^e -> lr(0)=2, after one step lr=2*0.5=1.
        let mut s = LambdaLR::new(2.0, |e| 0.5_f32.powi(e as i32));
        assert!((s.lr() - 2.0).abs() < 1e-6);
        assert!((s.step() - 1.0).abs() < 1e-6);
    }
}
