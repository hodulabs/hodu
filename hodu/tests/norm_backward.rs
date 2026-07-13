//! Backward wiring for the normalization layers: a real gradient must reach the learned
//! gamma/beta affine, so a few optimizer steps fit an all-ones target (reachable via
//! beta=mean, gamma=scale) -- loss drops and every norm param moves. The engine norm ops
//! are oracle-verified; this pins the hodu-side channel_affine composition.
use hodu::prelude::*;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 0.5).collect()
}

fn assert_norm_trains(m: &dyn Module, x: &Tensor) {
    let y = m.forward(x).unwrap();
    let n: usize = y.shape().iter().product();
    let target = x.ctx().constant(vec![1.0; n], y.shape().to_vec()); // not zero-mean -> beta must move
    let loss = mse_loss(&y, &target).unwrap();
    let params = m.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let before: Vec<Vec<f32>> = params.iter().map(Param::value).collect();
    let mut opt = Adam::new(params.clone(), 0.05);
    let l0 = loss.item();
    for _ in 0..80 {
        opt.step(&grad_values(&grads));
    }
    assert!(loss.item() < l0 * 0.5, "loss did not drop: {l0} -> {}", loss.item());
    for (p, b0) in params.iter().zip(&before) {
        assert!(p.value().iter().zip(b0).any(|(a, b)| (a - b).abs() > 1e-5), "a norm param did not move");
    }
}

#[test]
fn group_norm_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(4 * 2 * 2), vec![1, 4, 2, 2]);
    assert_norm_trains(&GroupNorm::new(&ctx, 4, 2, 1e-5), &x);
}

#[test]
fn instance_norm_backward() {
    let ctx = Ctx::cpu();
    let x = ctx.constant(ramp(3 * 4), vec![1, 3, 4]);
    assert_norm_trains(&InstanceNorm::new(&ctx, 3, 1e-5), &x);
}

#[test]
fn rms_norm_backward() {
    // single row so gamma alone can fit the ones target (output = gamma * x/rms).
    let ctx = Ctx::cpu();
    let x = ctx.constant(vec![1.0, -2.0, 3.0, -0.5], vec![1, 4]);
    assert_norm_trains(&RmsNorm::new(&ctx, 4, 1e-6), &x);
}
