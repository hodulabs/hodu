//! Direct 1-D convolution: Conv1d. Weight is `[O,C,K]` (out_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct Conv1d {
    w: Param,
    b: Option<Param>,
    out_ch: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*K`, from a
    /// deterministic `seed`. Dilation defaults to `1`; set it with [`Conv1d::with_dilation`].
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Conv1d {
        Conv1d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, seed, Init::HeUniform, true)
    }

    /// Same as [`Conv1d::new`], with a chosen weight initializer. `fan_in = C*K`,
    /// `fan_out = O*K`. `bias=false` drops the bias (no Param, no bias add).
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        seed: u64,
        init: Init,
        bias: bool,
    ) -> Conv1d {
        let k = kernel;
        let fan_in = in_ch * k;
        let fan_out = out_ch * k;
        let n = out_ch * fan_in;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        Conv1d {
            w: Param::new(ctx, w, vec![out_ch, in_ch, k]),
            b: bias.then(|| Param::new(ctx, vec![0.0; out_ch], vec![out_ch])),
            out_ch,
            stride,
            padding,
            dilation: 1,
        }
    }

    /// Set the dilation (spacing between kernel elements), default `1`. Dilation enlarges
    /// the effective kernel to `d*(K-1)+1`, shrinking the map.
    pub fn with_dilation(mut self, d: usize) -> Self {
        self.dilation = d;
        self
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv1d(self.w.tensor(), self.stride, self.padding, self.dilation)?;
        match &self.b {
            // bias [O] -> [1, O, 1] broadcasts over N, Lo.
            Some(b) => y.try_add(&b.tensor().reshape(vec![1, self.out_ch, 1])?),
            None => Ok(y),
        }
    }
    fn parameters(&self) -> Vec<Param> {
        let mut ps = vec![self.w.clone()];
        ps.extend(self.b.clone());
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_forward_shape() {
        // kernel 3 over a valid length-7 map (no padding, stride 1) -> length 5.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 7], vec![1, 3, 7]);
        let conv = Conv1d::new(&ctx, 3, 4, 3, 1, 0, 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 5]);
    }
}
