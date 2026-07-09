//! Dropout on the static graph: fresh mask per step, identity in eval, one graph.
use hodu::prelude::*;

#[test]
fn dropout_masks_train_fresh_and_eval_identity() {
    let ctx = Ctx::cpu();
    let n = 400;
    let x = ctx.constant(vec![1.0; n], vec![1, n]);
    let drop = Dropout::new(&ctx, 0.5).unwrap();
    let y = drop.forward(&x).unwrap(); // graph built ONCE

    // train mode: ~half zeroed, survivors scaled to 1/(1-0.5) = 2.
    ctx.set_training(true);
    let m1 = y.realize();
    let zeros1 = m1.iter().filter(|&&v| v == 0.0).count();
    assert!((150..=250).contains(&zeros1), "zeros {zeros1} not ~half");
    assert!(m1.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-6));

    // a fresh step draws a different mask.
    ctx.tick_rng();
    let m2 = y.realize();
    assert!(m1 != m2, "mask did not change across steps");

    // eval mode: identity (every input passes through unchanged).
    ctx.set_training(false);
    let e = y.realize();
    assert!(e.iter().all(|&v| (v - 1.0).abs() < 1e-6), "eval not identity");
}

#[test]
fn model_with_dropout_trains() {
    // regression y = sum(x); a net with dropout should still reduce its loss.
    let ctx = Ctx::cpu();
    let (b, din) = (16, 4);
    let xs: Vec<f32> = (0..b * din).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let ys: Vec<f32> = xs.chunks(din).map(|r| r.iter().sum()).collect();
    let x = ctx.input(vec![b, din]);
    let t = ctx.input(vec![b, 1]);
    ctx.feed(x.node(), xs, vec![b, din]);
    ctx.feed(t.node(), ys, vec![b, 1]);

    let net = Sequential::new(vec![
        Box::new(Linear::new(&ctx, din, 16, 1)),
        Box::new(Relu),
        Box::new(Dropout::new(&ctx, 0.2).unwrap()),
        Box::new(Linear::new(&ctx, 16, 1, 2)),
    ]);
    let loss = mse_loss(&net.forward(&x).unwrap(), &t).unwrap();
    let params = net.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.02);

    ctx.set_training(true);
    let l0 = loss.item();
    for _ in 0..80 {
        ctx.tick_rng();
        opt.step(&grad_values(&grads));
    }
    ctx.set_training(false);
    let lf = loss.item();
    assert!(lf < l0 * 0.5, "loss {l0:.4} -> {lf:.4} did not drop");
}

#[test]
fn dropout_rejects_p_out_of_range() {
    let ctx = Ctx::cpu();
    assert!(Dropout::new(&ctx, 1.0).is_err(), "p=1 -> inf scale, must Err");
    assert!(Dropout::new(&ctx, -0.1).is_err(), "p<0 is nonsense, must Err");
    assert!(Dropout::new(&ctx, 0.5).is_ok());
}
