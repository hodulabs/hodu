//! Transposed 3-D convolution: ConvTranspose3d. Weight is `[C,O,Kd,Kh,Kw]` (in_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct ConvTranspose3d {
    w: Param,
    b: Option<Param>,
    out_ch: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
}

impl ConvTranspose3d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*Kd*Kh*Kw`, from a
    /// deterministic `seed`. Weight is `[C, O, Kd, Kh, Kw]` (in_ch first). Dilation defaults
    /// to `(1, 1, 1)`; set it with [`ConvTranspose3d::with_dilation`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        seed: u64,
    ) -> ConvTranspose3d {
        ConvTranspose3d::with_init(
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

    /// Same as [`ConvTranspose3d::new`], with a chosen weight initializer.
    /// `fan_in = C*Kd*Kh*Kw`, `fan_out = O*Kd*Kh*Kw`. `bias=false` drops the bias (no Param,
    /// no bias add).
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        seed: u64,
        init: Init,
        bias: bool,
    ) -> ConvTranspose3d {
        let (kd, kh, kw) = kernel;
        let fan_in = in_ch * kd * kh * kw;
        let fan_out = out_ch * kd * kh * kw;
        let n = in_ch * out_ch * kd * kh * kw;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        ConvTranspose3d {
            w: Param::new(ctx, w, vec![in_ch, out_ch, kd, kh, kw]),
            b: bias.then(|| Param::new(ctx, vec![0.0; out_ch], vec![out_ch])),
            out_ch,
            stride,
            padding,
            output_padding,
            dilation: (1, 1, 1),
        }
    }

    /// Set the dilation (spacing between kernel elements), default `(1, 1, 1)`.
    pub fn with_dilation(mut self, dd: usize, dh: usize, dw: usize) -> Self {
        self.dilation = (dd, dh, dw);
        self
    }
}

impl Module for ConvTranspose3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv_transpose3d(self.w.tensor(), self.stride, self.padding, self.output_padding, self.dilation)?;
        match &self.b {
            // bias [O] -> [1, O, 1, 1, 1] broadcasts over N, Do, Ho, Wo.
            Some(b) => y.try_add(&b.tensor().reshape(vec![1, self.out_ch, 1, 1, 1])?),
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
    fn conv_transpose3d_upsamples() {
        // 4x4x4 -> (4-1)*1 + (3-1) + 1 = 6 per spatial axis.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 4 * 4 * 4], vec![1, 3, 4, 4, 4]);
        let conv = ConvTranspose3d::new(&ctx, 3, 4, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 6, 6, 6]);
    }
}
