use super::sched_state;

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
sched_state!(StepLR);

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
sched_state!(MultiStepLR);

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
}
