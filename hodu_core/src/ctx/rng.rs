//! Train/eval + dropout-seed plumbing for the build-once graph. One shared `flag`
//! Input (1.0 train / 0.0 eval) turns every dropout into identity in eval, and each
//! dropout's `seed` Input is refreshed per step (`tick_rng`) so the mask is fresh --
//! all without rebuilding the graph. State lives in `CtxInner.rng` (see ctx.rs).
use crate::{Ctx, Tensor};

impl Ctx {
    /// The shared train/eval flag Input (`1.0` train, `0.0` eval), created lazily.
    /// A `Dropout` reads it to become identity in eval with the SAME graph.
    pub fn train_flag(&self) -> Tensor {
        if let Some(n) = self.0.rng.borrow().flag {
            return self.wrap(n);
        }
        let t = self.input(vec![1]);
        let init = if self.0.rng.borrow().training { 1.0 } else { 0.0 };
        self.feed(t.node(), vec![init], vec![1]);
        self.0.rng.borrow_mut().flag = Some(t.node());
        t
    }

    /// Register a fresh per-dropout seed Input (I64 scalar), fed by [`Ctx::tick_rng`].
    pub fn new_dropout_seed(&self) -> Tensor {
        let t = self.input_i64(vec![1]);
        let idx = self.0.rng.borrow().seeds.len() as i64;
        self.feed_i64(t.node(), vec![idx], vec![1]); // counter 0
        self.0.rng.borrow_mut().seeds.push(t.node());
        t
    }

    /// Switch dropout between train (fresh masks) and eval (identity), by feeding
    /// the shared flag. Feed the batch, then realize -- one graph serves both.
    pub fn set_training(&self, training: bool) {
        self.0.rng.borrow_mut().training = training;
        let flag = self.0.rng.borrow().flag;
        if let Some(f) = flag {
            self.feed(f, vec![if training { 1.0 } else { 0.0 }], vec![1]);
        }
    }

    pub fn is_training(&self) -> bool {
        self.0.rng.borrow().training
    }

    /// Advance the dropout RNG one step: refeed every registered seed so the next
    /// realize draws a fresh mask. Call once per training step before `opt.step`.
    pub fn tick_rng(&self) {
        let (c, seeds) = {
            let mut r = self.0.rng.borrow_mut();
            r.counter = r.counter.wrapping_add(1);
            (r.counter, r.seeds.clone())
        };
        for (i, s) in seeds.iter().enumerate() {
            let seed = c.wrapping_mul(1_000_003).wrapping_add(i as u64) as i64;
            self.feed_i64(*s, vec![seed], vec![1]);
        }
    }
}
