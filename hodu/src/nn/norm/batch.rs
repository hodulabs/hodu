//! Batch normalization over `[N, C, ..]` (per-channel, axis 1). Composed from
//! reductions + the fed train/eval flag -- no engine op needed.
//!
//! A build-once graph can't swap subgraphs, so train (batch stats) vs
//! eval (running stats) is one graph blended by the shared `train_flag` (1.0 train
//! / 0.0 eval, like Dropout). The running mean/var are fed Inputs updated host-side:
//! call `update_running()` once per training step (after feeding the batch) to EMA
//! them from the current batch stats -- the static-graph analog of PyTorch's buffer
//! update. Variance is biased (divide by count) for both normalize and running;
//! PyTorch uses unbiased for the running buffer -- a documented parity gap.
//! Running stats are `Buffer`s (non-learnable, host-valued), reported via
//! `Module::buffers` so `save`/`load` persist them -- eval-mode inference is correct
//! after a round-trip.
use crate::nn::norm::channel_affine;
use crate::nn::{Buffer, Module, Param};
use hodu_core::{Ctx, Error, NodeId, Tensor};
use std::cell::Cell;

/// Batch norm for any `[N, C, ..]` (reduces every non-channel axis). `BatchNorm1d`
/// (`[N,C]`/`[N,C,L]`) and `BatchNorm2d` (`[N,C,H,W]`) are the same math.
pub struct BatchNorm {
    gamma: Param,
    beta: Param,
    running_mean: Buffer,             // fed Input [C]
    running_var: Buffer,              // fed Input [C]
    batch_mean: Cell<Option<NodeId>>, // set at forward, read by update_running
    batch_var: Cell<Option<NodeId>>,
    c: usize,
    eps: f32,
    momentum: f32,
}

impl BatchNorm {
    /// `c` channels, `eps` for numerical stability, `momentum` for the running-stat
    /// EMA (PyTorch default 0.1). `gamma`/`running_var` init 1, `beta`/`running_mean` init 0.
    pub fn new(ctx: &Ctx, c: usize, eps: f32, momentum: f32) -> BatchNorm {
        BatchNorm {
            gamma: Param::new(ctx, vec![1.0; c], vec![c]),
            beta: Param::new(ctx, vec![0.0; c], vec![c]),
            running_mean: Buffer::new(ctx, vec![0.0; c], vec![c]),
            running_var: Buffer::new(ctx, vec![1.0; c], vec![c]),
            batch_mean: Cell::new(None),
            batch_var: Cell::new(None),
            c,
            eps,
            momentum,
        }
    }

    /// EMA the running mean/var from the current batch's stats and re-feed them.
    /// Call once per training step, after the batch is fed (so the batch-stat nodes
    /// realize on the current data). No-op before the first `forward`.
    pub fn update_running(&self) {
        let (Some(bm), Some(bv)) = (self.batch_mean.get(), self.batch_var.get()) else {
            return;
        };
        let ctx = self.running_mean.tensor().ctx();
        let bmv = ctx.eval_f32(bm);
        let bvv = ctx.eval_f32(bv);
        let m = self.momentum;
        let (mut rm, mut rv) = (self.running_mean.value(), self.running_var.value());
        for i in 0..self.c {
            rm[i] = (1.0 - m) * rm[i] + m * bmv[i];
            rv[i] = (1.0 - m) * rv[i] + m * bvv[i];
        }
        self.running_mean.set(rm);
        self.running_var.set(rv);
    }
}

// reduce every axis except the channel (axis 1) -> [C]; descending so a dropped
// higher axis doesn't shift the lower ones still to reduce.
fn per_channel(x: &Tensor) -> Result<Tensor, Error> {
    let mut t = x.clone();
    for ax in (0..x.rank()).rev() {
        if ax != 1 {
            t = t.mean_axis(ax)?;
        }
    }
    Ok(t)
}

// [1, C, 1, ..] so a [C] stat broadcasts on the channel axis of a rank-`r` input.
fn bc_shape(r: usize, c: usize) -> Vec<usize> {
    let mut s = vec![1usize; r];
    s[1] = c;
    s
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        if x.rank() < 2 {
            return Err(Error::Shape {
                op: "BatchNorm",
                msg: format!("expected rank >= 2 input (channel dim present), got rank {}", x.rank()),
            });
        }
        let bc = bc_shape(x.rank(), self.c);
        // batch stats (biased), always computed in-graph
        let bmean = per_channel(x)?; // [C]
        let centered = x.try_sub(&bmean.reshape(bc.clone())?)?;
        let bvar = per_channel(&centered.square())?; // [C]
        self.batch_mean.set(Some(bmean.node()));
        self.batch_var.set(Some(bvar.node()));
        // blend by the fed flag: train -> batch stats, eval -> running stats
        let flag = x.ctx().train_flag(); // [1]
        let ev = &(&flag * -1.0) + 1.0; // eval weight = 1 - flag
        let mean = flag.try_mul(&bmean)?.try_add(&ev.try_mul(self.running_mean.tensor())?)?;
        let var = flag.try_mul(&bvar)?.try_add(&ev.try_mul(self.running_var.tensor())?)?;
        let denom = (&var.reshape(bc.clone())? + self.eps).sqrt();
        let norm = x.try_sub(&mean.reshape(bc)?)?.try_div(&denom)?;
        channel_affine(&norm, &self.gamma, &self.beta)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
    fn buffers(&self) -> Vec<Buffer> {
        vec![self.running_mean.clone(), self.running_var.clone()]
    }
}

// BatchNorm1d/2d wrap the general `BatchNorm` and reject any input whose rank isn't the
// PyTorch-style expected one, so a `[N,C,H,W]` fed to a `BatchNorm1d` (or a `[N,C]` to a
// `BatchNorm2d`) Errs at the layer instead of silently running the wrong-rank reduction.
macro_rules! ranked_batchnorm {
    ($name:ident, $doc:literal, $ranks:pat, $want:literal) => {
        #[doc = $doc]
        pub struct $name(BatchNorm);

        impl $name {
            /// See [`BatchNorm::new`].
            pub fn new(ctx: &Ctx, c: usize, eps: f32, momentum: f32) -> $name {
                $name(BatchNorm::new(ctx, c, eps, momentum))
            }
            /// See [`BatchNorm::update_running`].
            pub fn update_running(&self) {
                self.0.update_running();
            }
        }

        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
                if !matches!(x.rank(), $ranks) {
                    return Err(Error::Shape {
                        op: stringify!($name),
                        msg: format!(concat!("expected ", $want, " input, got rank {}"), x.rank()),
                    });
                }
                self.0.forward(x)
            }
            fn parameters(&self) -> Vec<Param> {
                self.0.parameters()
            }
            fn buffers(&self) -> Vec<Buffer> {
                self.0.buffers()
            }
        }
    };
}

ranked_batchnorm!(
    BatchNorm1d,
    "BatchNorm for `[N, C]` / `[N, C, L]` inputs (PyTorch `BatchNorm1d`; any other rank Errs).",
    2 | 3,
    "[N,C] or [N,C,L]"
);
ranked_batchnorm!(
    BatchNorm2d,
    "BatchNorm for `[N, C, H, W]` inputs (PyTorch `BatchNorm2d`; any other rank Errs).",
    4,
    "[N,C,H,W]"
);
