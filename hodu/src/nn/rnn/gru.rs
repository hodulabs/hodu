//! An unrolled GRU cell (reset/update gates).
use crate::nn::{Module, Param, uniform};
use hodu_core::{Ctx, Error, Tensor};

/// An unrolled GRU (reset/update gates). Two bias vectors are kept because the reset
/// gate multiplies the hidden projection BEFORE the input projection is added, so the
/// ih/hh biases cannot be folded together: `n = tanh(x@W_in + r*(h@W_hn))`. Weights
/// `w_ih [in, 3H]`, `w_hh [H, 3H]` pack the reset/update/new gates; `b_ih`/`b_hh` are
/// `[3H]`. Returns the last hidden `[B,H]`.
pub struct Gru {
    w_ih: Param,
    w_hh: Param,
    b_ih: Param,
    b_hh: Param,
    hidden: usize,
}

impl Gru {
    /// He-uniform init, zero biases, deterministic `seed`.
    pub fn new(ctx: &Ctx, in_features: usize, hidden: usize, seed: u64) -> Gru {
        let g = 3 * hidden;
        let ib = 1.0 / (in_features as f32).sqrt();
        let hb = 1.0 / (hidden as f32).sqrt();
        Gru {
            w_ih: Param::new(ctx, uniform(in_features * g, ib, seed), vec![in_features, g]),
            w_hh: Param::new(ctx, uniform(hidden * g, hb, seed ^ 0x2468), vec![hidden, g]),
            b_ih: Param::new(ctx, vec![0.0; g], vec![g]),
            b_hh: Param::new(ctx, vec![0.0; g], vec![g]),
            hidden,
        }
    }
}

impl Module for Gru {
    /// `x` `[B,T,in]` -> last hidden `[B,H]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let ctx = x.ctx();
        let (b, t, inf) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hn = self.hidden;
        let (w_ih, w_hh) = (self.w_ih.tensor(), self.w_hh.tensor());
        let (b_ih, b_hh) = (self.b_ih.tensor(), self.b_hh.tensor());

        let mut h = ctx.zeros(vec![b, hn]);
        for step in 0..t {
            let xt = x.slice(vec![(0, b), (step, step + 1), (0, inf)])?.squeeze(1)?;
            let gi = &xt.matmul(w_ih)? + b_ih; // input projection [B,3H]
            let gh = &h.matmul(w_hh)? + b_hh; // hidden projection [B,3H]
            let (ir, iz, in_) = split3(&gi, hn)?;
            let (hr, hz, hn_) = split3(&gh, hn)?;
            let r = (&ir + &hr).sigmoid();
            let z = (&iz + &hz).sigmoid();
            let n = (&in_ + &(&r * &hn_)).tanh(); // reset gates the hidden projection
            // h_t = (1-z)*n + z*h
            let one = h.scalar_like(1.0);
            h = &(&(&one - &z) * &n) + &(&z * &h);
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.b_ih.clone(), self.b_hh.clone()]
    }
}

// split a [B,3H] gate matrix into its three [B,H] parts.
fn split3(g: &Tensor, h: usize) -> Result<(Tensor, Tensor, Tensor), Error> {
    let mut parts = g.split(&[h, h, h], 1)?;
    let c = parts.remove(2);
    let b = parts.remove(1);
    let a = parts.remove(0);
    Ok((a, b, c))
}
