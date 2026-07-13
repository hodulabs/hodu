//! Backward wiring for the conv layers (forward shape is covered by unit tests): a real
//! gradient must reach the bias through its `[1,O,1..]` broadcast, and a few optimizer
//! steps must reduce an mse-to-zero loss. The engine conv ops are oracle-verified; this
//! pins the hodu-side bias-add composition, not the convolution math.
use hodu::prelude::*;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 0.5).collect()
}

// bias grad must be nonzero (broadcast wiring intact) and training must drop the loss.
fn assert_conv_trains(m: &dyn Module, x: &Tensor) {
    let y = m.forward(x).unwrap();
    let target = x.ctx().zeros(y.shape().to_vec());
    let loss = mse_loss(&y, &target).unwrap();
    let params = m.parameters(); // [w, b]
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let g0 = grad_values(&grads);
    assert!(g0[1].iter().any(|&v| v.abs() > 1e-6), "bias grad is zero -> broadcast wiring broken");
    let mut opt = Adam::new(params.clone(), 0.05);
    let l0 = loss.item();
    for _ in 0..80 {
        opt.step(&grad_values(&grads));
    }
    assert!(loss.item() < l0 * 0.5, "loss did not drop: {l0} -> {}", loss.item());
}

#[test]
fn conv1d_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(2 * 5), vec![1, 2, 5]);
    assert_conv_trains(&Conv1d::new(&ctx, 2, 3, 3, 1, 1, 7), &x);
}

#[test]
fn conv3d_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(2 * 4 * 4 * 4), vec![1, 2, 4, 4, 4]);
    assert_conv_trains(&Conv3d::new(&ctx, 2, 3, (2, 2, 2), (1, 1, 1), (0, 0, 0), 7), &x);
}

#[test]
fn conv_transpose1d_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(2 * 5), vec![1, 2, 5]);
    assert_conv_trains(&ConvTranspose1d::new(&ctx, 2, 3, 3, 1, 0, 0, 7), &x);
}

#[test]
fn conv_transpose2d_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(2 * 5 * 5), vec![1, 2, 5, 5]);
    assert_conv_trains(&ConvTranspose2d::new(&ctx, 2, 3, (3, 3), (1, 1), (0, 0), (0, 0), 7), &x);
}

#[test]
fn conv_transpose3d_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(2 * 4 * 4 * 4), vec![1, 2, 4, 4, 4]);
    assert_conv_trains(&ConvTranspose3d::new(&ctx, 2, 3, (2, 2, 2), (1, 1, 1), (0, 0, 0), (0, 0, 0), 7), &x);
}
