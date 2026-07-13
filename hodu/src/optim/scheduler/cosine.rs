use super::sched_state;

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
sched_state!(CosineAnnealingLR);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_starts_at_base_hits_eta_min_at_tmax() {
        let mut s = CosineAnnealingLR::new(1.0, 4, 0.1);
        assert!((s.lr() - 1.0).abs() < 1e-6); // epoch 0: eta_min + (base-eta_min)*1 = base
        for _ in 0..4 {
            s.step();
        }
        assert!((s.lr() - 0.1).abs() < 1e-6, "lr at t_max {}", s.lr()); // anneals to eta_min
    }
}
