//! Direct convolution: Conv1d/Conv2d/Conv3d. Weight is `[O,C,K...]` (out_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct Conv1d {
    w: Param,
    b: Param,
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
        Conv1d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, seed, Init::HeUniform)
    }

    /// Same as [`Conv1d::new`], with a chosen weight initializer. `fan_in = C*K`,
    /// `fan_out = O*K`.
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
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
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
        // bias [O] -> [1, O, 1] broadcasts over N, Lo.
        let b = self.b.tensor().reshape(vec![1, self.out_ch, 1])?;
        y.try_add(&b)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub struct Conv2d {
    w: Param,
    b: Param,
    out_ch: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl Conv2d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*KH*KW`,
    /// from a deterministic `seed`. Dilation defaults to `(1, 1)`; set it with
    /// [`Conv2d::with_dilation`].
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        seed: u64,
    ) -> Conv2d {
        Conv2d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, seed, Init::HeUniform)
    }

    /// Same as [`Conv2d::new`], with a chosen weight initializer. `fan_in = C*KH*KW`,
    /// `fan_out = O*KH*KW`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        seed: u64,
        init: Init,
    ) -> Conv2d {
        let (kh, kw) = kernel;
        let fan_in = in_ch * kh * kw;
        let fan_out = out_ch * kh * kw;
        let n = out_ch * fan_in;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        Conv2d {
            w: Param::new(ctx, w, vec![out_ch, in_ch, kh, kw]),
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
            out_ch,
            stride,
            padding,
            dilation: (1, 1),
        }
    }

    /// Set the dilation (spacing between kernel elements), default `(1, 1)`. Dilation
    /// enlarges the effective kernel to `(dh*(KH-1)+1, dw*(KW-1)+1)`, shrinking the map.
    pub fn with_dilation(mut self, dh: usize, dw: usize) -> Self {
        self.dilation = (dh, dw);
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv2d(self.w.tensor(), self.stride, self.padding, self.dilation)?;
        // bias [O] -> [1, O, 1, 1] broadcasts over N, Ho, Wo.
        let b = self.b.tensor().reshape(vec![1, self.out_ch, 1, 1])?;
        y.try_add(&b)
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub struct Conv3d {
    w: Param,
    b: Param,
    out_ch: usize,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
}

impl Conv3d {
    /// He-uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, `fan_in = C*Kd*Kh*Kw`,
    /// from a deterministic `seed`. Dilation defaults to `(1, 1, 1)`; set it with
    /// [`Conv3d::with_dilation`].
    pub fn new(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        seed: u64,
    ) -> Conv3d {
        Conv3d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, seed, Init::HeUniform)
    }

    /// Same as [`Conv3d::new`], with a chosen weight initializer. `fan_in = C*Kd*Kh*Kw`,
    /// `fan_out = O*Kd*Kh*Kw`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_init(
        ctx: &Ctx,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        seed: u64,
        init: Init,
    ) -> Conv3d {
        let (kd, kh, kw) = kernel;
        let fan_in = in_ch * kd * kh * kw;
        let fan_out = out_ch * kd * kh * kw;
        let n = out_ch * fan_in;
        let w = match init {
            Init::HeUniform => uniform(n, 1.0 / (fan_in as f32).sqrt(), seed),
            Init::XavierUniform => xavier_uniform(n, fan_in, fan_out, seed),
            Init::KaimingNormal => kaiming_normal(n, fan_in, seed),
            Init::Normal(std) => normal(n, std, seed),
            Init::XavierNormal => xavier_normal(n, fan_in, fan_out, seed),
        };
        Conv3d {
            w: Param::new(ctx, w, vec![out_ch, in_ch, kd, kh, kw]),
            b: Param::new(ctx, vec![0.0; out_ch], vec![out_ch]),
            out_ch,
            stride,
            padding,
            dilation: (1, 1, 1),
        }
    }

    /// Set the dilation (spacing between kernel elements), default `(1, 1, 1)`. Dilation
    /// enlarges the effective kernel to `(dd*(Kd-1)+1, dh*(Kh-1)+1, dw*(Kw-1)+1)`.
    pub fn with_dilation(mut self, dd: usize, dh: usize, dw: usize) -> Self {
        self.dilation = (dd, dh, dw);
        self
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = x.conv3d(self.w.tensor(), self.stride, self.padding, self.dilation)?;
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
    fn init_changes_weights() {
        let ctx = Ctx::cpu();
        let default = Conv2d::new(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7);
        let xavier = Conv2d::with_init(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7, Init::XavierUniform);
        assert_ne!(default.w.value(), xavier.w.value(), "a non-default init must change the initial weights");
        // new defaults to He-uniform: same seed + HeUniform reproduces `new` exactly.
        let he = Conv2d::with_init(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7, Init::HeUniform);
        assert_eq!(default.w.value(), he.w.value());
    }

    #[test]
    fn with_dilation_enlarges_effective_kernel() {
        // 3x3 kernel over a valid 7x7 map (no padding, stride 1).
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 7 * 7], vec![1, 3, 7, 7]);
        // dilation 1: effective kernel 3 -> out 5x5.
        let plain = Conv2d::new(&ctx, 3, 4, (3, 3), (1, 1), (0, 0), 0);
        assert_eq!(plain.forward(&x).unwrap().shape(), &[1, 4, 5, 5]);
        // dilation 2: effective kernel 2*(3-1)+1 = 5 -> out 3x3.
        let dil = Conv2d::new(&ctx, 3, 4, (3, 3), (1, 1), (0, 0), 0).with_dilation(2, 2);
        assert_eq!(dil.forward(&x).unwrap().shape(), &[1, 4, 3, 3]);
    }

    #[test]
    fn conv1d_forward_shape() {
        // kernel 3 over a valid length-7 map (no padding, stride 1) -> length 5.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 7], vec![1, 3, 7]);
        let conv = Conv1d::new(&ctx, 3, 4, 3, 1, 0, 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 5]);
    }

    #[test]
    fn conv3d_forward_shape() {
        // 3x3x3 kernel over a valid 5x5x5 map (no padding, stride 1) -> 3x3x3.
        let ctx = Ctx::cpu();
        let x = ctx.constant(vec![0.0; 3 * 5 * 5 * 5], vec![1, 3, 5, 5, 5]);
        let conv = Conv3d::new(&ctx, 3, 4, (3, 3, 3), (1, 1, 1), (0, 0, 0), 0);
        assert_eq!(conv.forward(&x).unwrap().shape(), &[1, 4, 3, 3, 3]);
    }
}
