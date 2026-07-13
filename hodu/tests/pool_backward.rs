//! Backward wiring for the (parameter-free) pooling layers: a learnable input tensor is
//! pooled, and the pool must pass a real gradient back to it -- nonzero input grad plus a
//! dropping mse-to-zero loss over a few optimizer steps. The engine pool ops are
//! oracle-verified; this pins that the layer composes into a differentiable graph.
use hodu::prelude::*;

fn ramp(n: usize) -> Vec<f32> {
    // distinct values so max-pool has a well-defined argmax per window.
    (0..n).map(|i| (i as f32) * 0.13 + 0.5).collect()
}

fn assert_pool_trains(m: &dyn Module, w: &Param) {
    let y = m.forward(w.tensor()).unwrap();
    let target = w.tensor().ctx().zeros(y.shape().to_vec());
    let loss = mse_loss(&y, &target).unwrap();
    let grads = loss.grad(&[w.tensor()]).unwrap();
    let g0 = grad_values(&grads);
    assert!(g0[0].iter().any(|&v| v.abs() > 1e-6), "pool passed no gradient to its input");
    let mut opt = Adam::new(vec![w.clone()], 0.1);
    let l0 = loss.item();
    for _ in 0..80 {
        opt.step(&grad_values(&grads));
    }
    assert!(loss.item() < l0 * 0.5, "loss did not drop: {l0} -> {}", loss.item());
}

#[test]
fn avg_pool1d_backward() {
    let ctx = Ctx::cpu();
    let w = Param::new(&ctx, ramp(6), vec![1, 1, 6]);
    assert_pool_trains(&AvgPool1d::new(2, 2), &w);
}

#[test]
fn avg_pool2d_backward() {
    let ctx = Ctx::cpu();
    let w = Param::new(&ctx, ramp(16), vec![1, 1, 4, 4]);
    assert_pool_trains(&AvgPool2d::new((2, 2), (2, 2)), &w);
}

#[test]
fn avg_pool3d_backward() {
    let ctx = Ctx::cpu();
    let w = Param::new(&ctx, ramp(64), vec![1, 1, 4, 4, 4]);
    assert_pool_trains(&AvgPool3d::new((2, 2, 2), (2, 2, 2)), &w);
}

#[test]
fn max_pool1d_backward() {
    let ctx = Ctx::cpu();
    let w = Param::new(&ctx, ramp(6), vec![1, 1, 6]);
    assert_pool_trains(&MaxPool1d::new(2, 2), &w);
}

#[test]
fn max_pool3d_backward() {
    let ctx = Ctx::cpu();
    let w = Param::new(&ctx, ramp(64), vec![1, 1, 4, 4, 4]);
    assert_pool_trains(&MaxPool3d::new((2, 2, 2), (2, 2, 2)), &w);
}
