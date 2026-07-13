//! Transposed convolution: ConvTranspose1d/2d/3d. Weight is `[C,O,K...]` (in_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct ConvTranspose1d {
    w: Param,
    b: Param,
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
        ConvTranspose1d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, output_padding, seed, Init::HeUniform)
    }

    /// Same as [`ConvTranspose1d::new`], with a chosen weight initializer. `fan_in = C*K`,
    /// `fan_out = O*K`.
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
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
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
        // bias [O] -> [1, O, 1] broadcasts over N, Lo.
        let b = self.b.tensor().reshape(vec![1, self.out_ch, 1])?;
        y.try_add(&b)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub struct ConvTranspose2d {
    w: Param,
    b: Param,
    out_ch: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
}

impl ConvTranspose2d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*KH*KW`, from a
    /// deterministic `seed`. Weight is `[C, O, KH, KW]` (in_ch first). Dilation defaults to
    /// `(1, 1)`; set it with [`ConvTranspose2d::with_dilation`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        seed: u64,
    ) -> ConvTranspose2d {
        ConvTranspose2d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, output_padding, seed, Init::HeUniform)
    }

    /// Same as [`ConvTranspose2d::new`], with a chosen weight initializer. `fan_in = C*KH*KW`,
    /// `fan_out = O*KH*KW`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        seed: u64,
        init: Init,
    ) -> ConvTranspose2d {
        let (kh, kw) = kernel;
        let fan_in = in_ch * kh * kw;
        let fan_out = out_ch * kh * kw;
        let n = in_ch * out_ch * kh * kw;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        ConvTranspose2d {
            w: Param::new(ctx, w, vec![in_ch, out_ch, kh, kw]),
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
            out_ch,
            stride,
            padding,
            output_padding,
            dilation: (1, 1),
        }
    }

    /// Set the dilation (spacing between kernel elements), default `(1, 1)`.
    pub fn with_dilation(mut self, dh: usize, dw: usize) -> Self {
        self.dilation = (dh, dw);
        self
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv_transpose2d(self.w.tensor(), self.stride, self.padding, self.output_padding, self.dilation)?;
        // bias [O] -> [1, O, 1, 1] broadcasts over N, Ho, Wo.
        let b = self.b.tensor().reshape(vec![1, self.out_ch, 1, 1])?;
        y.try_add(&b)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub struct ConvTranspose3d {
    w: Param,
    b: Param,
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
        ConvTranspose3d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, output_padding, seed, Init::HeUniform)
    }

    /// Same as [`ConvTranspose3d::new`], with a chosen weight initializer.
    /// `fan_in = C*Kd*Kh*Kw`, `fan_out = O*Kd*Kh*Kw`.
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
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
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
        // bias [O] -> [1, O, 1, 1, 1] broadcasts over N, Do, Ho, Wo.
        let b = self.b.tensor().reshape(vec![1, self.out_ch, 1, 1, 1])?;
        y.try_add(&b)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
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

    #[test]
    fn conv_transpose2d_upsamples() {
        // stride 2 grows the map: 4x4 -> (4-1)*2 + (3-1) + 1 = 9x9.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 4 * 4], vec![1, 3, 4, 4]);
        let conv = ConvTranspose2d::new(&ctx, 3, 4, (3, 3), (2, 2), (0, 0), (0, 0), 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 9, 9]);
    }

    #[test]
    fn conv_transpose3d_upsamples() {
        // 4x4x4 -> (4-1)*1 + (3-1) + 1 = 6 per spatial axis.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 4 * 4 * 4], vec![1, 3, 4, 4, 4]);
        let conv = ConvTranspose3d::new(&ctx, 3, 4, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 6, 6, 6]);
    }
}
