//! 2-D convolution layer: `[N,C,H,W] -> [N,O,Ho,Wo]`, weight `[O,C,KH,KW]` plus a
//! per-output-channel bias broadcast over the spatial map.
use crate::nn::{Init, Module, Param, kaiming_normal, uniform, xavier_uniform};
use hodu_core::{Ctx, Error, Tensor};

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
    /// from a deterministic `seed`. Dilation is 1.
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
}
