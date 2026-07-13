use super::sched_state;

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
sched_state!(LambdaLR);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambdalr_applies_fn() {
        // base 2.0, f(e)=0.5^e -> lr(0)=2, after one step lr=2*0.5=1.
        let mut s = LambdaLR::new(2.0, |e| 0.5_f32.powi(e as i32));
        assert!((s.lr() - 2.0).abs() < 1e-6);
        assert!((s.step() - 1.0).abs() < 1e-6);
    }
}
