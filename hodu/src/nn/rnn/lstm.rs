//! An unrolled LSTM cell (input/forget/cell/output gates).
use crate::nn::{Module, Param, uniform};
use hodu_core::{Ctx, Error, Tensor};

/// An unrolled LSTM. The input-hidden `w_ih [in, 4H]`, hidden-hidden `w_hh [H, 4H]`
/// and `bias [4H]` pack the input/forget/cell/output gates side by side (one matmul
/// each, then `split`). `forward(x)` with `x` `[B,T,in]` runs the cell for `t in
/// 0..T`, carrying hidden `h` and cell `c`; by default it returns the LAST hidden
/// `[B,H]` (for classification), or the full sequence `[B,T,H]` when built with
/// [`Lstm::return_sequences`].
pub struct Lstm {
    w_ih: Param,
    w_hh: Param,
    bias: Param,
    hidden: usize,
    return_sequences: bool,
}

impl Lstm {
    /// He-uniform init (each weight in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`), zero
    /// bias, from a deterministic `seed`. Returns the last hidden state by default.
    pub fn new(ctx: &Ctx, in_features: usize, hidden: usize, seed: u64) -> Lstm {
        let g = 4 * hidden;
        let ib = 1.0 / (in_features as f32).sqrt();
        let hb = 1.0 / (hidden as f32).sqrt();
        Lstm {
            w_ih: Param::new(ctx, uniform(in_features * g, ib, seed), vec![in_features, g]),
            w_hh: Param::new(ctx, uniform(hidden * g, hb, seed ^ 0x1357), vec![hidden, g]),
            bias: Param::new(ctx, vec![0.0; g], vec![g]),
            hidden,
            return_sequences: false,
        }
    }

    /// Return the full output sequence `[B,T,H]` (stacked per-timestep hiddens)
    /// instead of just the last hidden -- for stacking LSTMs or seq2seq.
    pub fn return_sequences(mut self) -> Lstm {
        self.return_sequences = true;
        self
    }
}

impl Module for Lstm {
    /// `x` `[B,T,in]` -> last hidden `[B,H]` (default) or sequence `[B,T,H]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        if x.rank() != 3 {
            return Err(Error::Shape { op: "Lstm", msg: format!("expected [B,T,in] input, got rank {}", x.rank()) });
        }
        let ctx = x.ctx();
        let (b, t, inf) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hn = self.hidden;
        let (w_ih, w_hh, bias) = (self.w_ih.tensor(), self.w_hh.tensor(), self.bias.tensor());

        let mut h = ctx.zeros(vec![b, hn]);
        let mut c = ctx.zeros(vec![b, hn]);
        let mut seq: Vec<Tensor> = Vec::with_capacity(t);
        for step in 0..t {
            let xt = x.slice(vec![(0, b), (step, step + 1), (0, inf)])?.squeeze(1)?;
            // fused gate pre-activations [B,4H] = x_t @ w_ih + h @ w_hh + bias
            let gates = &(&xt.matmul(w_ih)? + &h.matmul(w_hh)?) + bias;
            let g = gates.split(&[hn, hn, hn, hn], 1)?;
            let (i, f, gg, o) = (g[0].sigmoid(), g[1].sigmoid(), g[2].tanh(), g[3].sigmoid());
            c = &(&f * &c) + &(&i * &gg); // c_t = f*c_{t-1} + i*g
            h = &o * &c.tanh(); // h_t = o*tanh(c_t)
            if self.return_sequences {
                seq.push(h.clone());
            }
        }
        if self.return_sequences {
            let refs: Vec<&Tensor> = seq.iter().collect();
            ctx.stack(&refs, 1)
        } else {
            Ok(h)
        }
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.bias.clone()]
    }
}
