//! Transposed 1-D convolution: ConvTranspose1d. Weight is `[C,O,K]` (in_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct ConvTranspose1d {
    w: Param,
    b: Option<Param>,
    out_ch: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
}

impl ConvTranspose1d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*K`, from a
    /// deterministic `seed`. Weight is `[C, O, K]` (in_ch first). Dilation defaults to `1`;
    /// set it with [`ConvTranspose1d::with_dilation`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        seed: u64,
    ) -> ConvTranspose1d {
        ConvTranspose1d::with_init(
            ctx,
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            output_padding,
            seed,
            Init::HeUniform,
            true,
        )
    }

    /// Same as [`ConvTranspose1d::new`], with a chosen weight initializer. `fan_in = C*K`,
    /// `fan_out = O*K`. `bias=false` drops the bias (no Param, no bias add).
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        seed: u64,
        init: Init,
        bias: bool,
    ) -> ConvTranspose1d {
        let k = kernel;
        let fan_in = in_ch * k;
        let fan_out = out_ch * k;
        let n = in_ch * out_ch * k;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        ConvTranspose1d {
            w: Param::new(ctx, w, vec![in_ch, out_ch, k]),
            b: bias.then(|| Param::new(ctx, vec![0.0; out_ch], vec![out_ch])),
            out_ch,
            stride,
            padding,
            output_padding,
            dilation: 1,
        }
    }

    /// Set the dilation (spacing between kernel elements), default `1`.
    pub fn with_dilation(mut self, d: usize) -> Self {
        self.dilation = d;
        self
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv_transpose1d(self.w.tensor(), self.stride, self.padding, self.output_padding, self.dilation)?;
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
    fn conv_transpose1d_upsamples() {
        // length 5 -> (5-1)*1 + (3-1) + 1 = 7.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 5], vec![1, 3, 5]);
        let conv = ConvTranspose1d::new(&ctx, 3, 4, 3, 1, 0, 0, 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 7]);
    }
}
