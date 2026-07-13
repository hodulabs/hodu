//! LR schedulers. Each is a pure LR curve over an epoch counter: call `step()` once
//! per epoch and hand the result to `opt.set_lr(..)` -- no borrow of the optimizer,
//! which the static `Cell`/`&mut` lr setters make unnecessary.

mod cosine;
mod lambda;
mod step;

pub use cosine::CosineAnnealingLR;
pub use lambda::LambdaLR;
pub use step::{MultiStepLR, StepLR};

/// A scheduler's resumable epoch counter, so `save_checkpoint`/`load_checkpoint` can
/// persist and restore where a run's LR curve was (the `sched.last_epoch` row).
pub trait SchedState {
    fn last_epoch(&self) -> usize;
    fn set_last_epoch(&mut self, epoch: usize);
}

// last_epoch is each scheduler's `epoch` field -- the only resumable state. Invoked
// beside each struct so the impl sees the private `epoch`; expands at each direct
// child of this module, so `super::SchedState` resolves back here.
macro_rules! sched_state {
    ($t:ty) => {
        impl super::SchedState for $t {
            fn last_epoch(&self) -> usize {
                self.epoch
            }
            fn set_last_epoch(&mut self, epoch: usize) {
                self.epoch = epoch;
            }
        }
    };
}
pub(crate) use sched_state;

#[cfg(test)]
mod tests {
    use super::*;

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
}
