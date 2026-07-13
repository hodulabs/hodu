//! Direct 2-D convolution: Conv2d. Weight is `[O,C,KH,KW]` (out_ch first).
use crate::nn::{Init, Module, Param, kaiming_normal, normal, uniform, xavier_normal, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

pub struct Conv2d {
    w: Param,
    b: Option<Param>,
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
        Conv2d::with_init(ctx, in_ch, out_ch, kernel, stride, padding, seed, Init::HeUniform, true)
    }

    /// Same as [`Conv2d::new`], with a chosen weight initializer. `fan_in = C*KH*KW`,
    /// `fan_out = O*KH*KW`. `bias=false` drops the bias (no Param, no bias add).
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
        bias: bool,
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
            b: bias.then(|| Param::new(ctx, vec![0.0; out_ch], vec![out_ch])),
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
        match &self.b {
            // bias [O] -> [1, O, 1, 1] broadcasts over N, Ho, Wo.
            Some(b) => y.try_add(&b.tensor().reshape(vec![1, self.out_ch, 1, 1])?),
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
    fn init_changes_weights() {
        let ctx = Ctx::cpu();
        let default = Conv2d::new(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7);
        let xavier = Conv2d::with_init(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7, Init::XavierUniform, true);
        assert_ne!(default.w.value(), xavier.w.value(), "a non-default init must change the initial weights");
        // new defaults to He-uniform: same seed + HeUniform reproduces `new` exactly.
        let he = Conv2d::with_init(&ctx, 3, 4, (3, 3), (1, 1), (1, 1), 7, Init::HeUniform, true);
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
}
