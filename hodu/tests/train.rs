//! End-to-end: the static frontend trains a linear regression to convergence.
use hodu::prelude::*;

#[test]
fn linear_regression_converges() {
    let ctx = Ctx::cpu();
    let n = 64usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 4.0 - 2.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x - 2.0).collect();

    let x = ctx.input(vec![n, 1]);
    ctx.feed(x.node(), xs, vec![n, 1]);
    let y = ctx.input(vec![n, 1]);
    ctx.feed(y.node(), ys, vec![n, 1]);

    let lin = Linear::new(&ctx, 1, 1, 0);
    let loss = (&lin.forward(&x).unwrap() - &y).square().mean_all().unwrap();

    let params = lin.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let sgd = Sgd::new(params.clone(), 0.2);

    for _ in 0..300 {
        sgd.step(&grad_values(&grads));
    }

    assert!(loss.item() < 1e-3, "loss did not converge: {}", loss.item());
    assert!((params[0].value()[0] - 3.0).abs() < 1e-2, "w = {}", params[0].value()[0]);
    assert!((params[1].value()[0] + 2.0).abs() < 1e-2, "b = {}", params[1].value()[0]);
}

#[test]
fn clipped_grads_still_converge() {
    // Same regression, but clip the global grad norm to 0.5 before each step. The
    // clip must fire on the large early grads, and training still converges.
    let ctx = Ctx::cpu();
    let n = 64usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 4.0 - 2.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x - 2.0).collect();

    let x = ctx.input(vec![n, 1]);
    ctx.feed(x.node(), xs, vec![n, 1]);
    let y = ctx.input(vec![n, 1]);
    ctx.feed(y.node(), ys, vec![n, 1]);

    let lin = Linear::new(&ctx, 1, 1, 0);
    let loss = (&lin.forward(&x).unwrap() - &y).square().mean_all().unwrap();
    let params = lin.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let sgd = Sgd::new(params.clone(), 0.2);

    let mut clip_fired = false;
    for _ in 0..500 {
        let mut gv = grad_values(&grads);
        if clip_grad_norm(&mut gv, 0.5) > 0.5 {
            clip_fired = true;
        }
        sgd.step(&gv);
    }
    assert!(clip_fired, "clip never fired -- grads never exceeded max_norm");
    assert!(loss.item() < 1e-2, "clipped run did not converge: {}", loss.item());
}

#[test]
fn grad_accumulation_matches_full_batch() {
    // Two micro-batches (each half the data), grads accumulated + averaged, one step:
    // for an MSE mean loss the averaged half-batch grads equal the full-batch grad, so
    // the accumulated run must track a full-batch run step for step.
    let n = 64usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 4.0 - 2.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x - 2.0).collect();
    let half = n / 2;

    // reference: one full-batch graph.
    let rctx = Ctx::cpu();
    let rx = rctx.input(vec![n, 1]);
    rctx.feed(rx.node(), xs.clone(), vec![n, 1]);
    let ry = rctx.input(vec![n, 1]);
    rctx.feed(ry.node(), ys.clone(), vec![n, 1]);
    let rlin = Linear::new(&rctx, 1, 1, 0);
    let rloss = (&rlin.forward(&rx).unwrap() - &ry).square().mean_all().unwrap();
    let rparams = rlin.parameters();
    let rpts: Vec<&Tensor> = rparams.iter().map(Param::tensor).collect();
    let rgrads = rloss.grad(&rpts).unwrap();
    let rsgd = Sgd::new(rparams.clone(), 0.2);
    for _ in 0..100 {
        rsgd.step(&grad_values(&rgrads));
    }
    let want: Vec<f32> = rparams.iter().flat_map(|p| p.value()).collect();

    // accumulated: one half-batch graph, fed each half then averaged before the step.
    let ctx = Ctx::cpu();
    let x = ctx.input(vec![half, 1]);
    let y = ctx.input(vec![half, 1]);
    let lin = Linear::new(&ctx, 1, 1, 0);
    let loss = (&lin.forward(&x).unwrap() - &y).square().mean_all().unwrap();
    let params = lin.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let sgd = Sgd::new(params.clone(), 0.2);
    for _ in 0..100 {
        let mut acc: Vec<Vec<f32>> = Vec::new();
        for h in 0..2 {
            ctx.feed(x.node(), xs[h * half..(h + 1) * half].to_vec(), vec![half, 1]);
            ctx.feed(y.node(), ys[h * half..(h + 1) * half].to_vec(), vec![half, 1]);
            accumulate_grads(&mut acc, &grad_values(&grads));
        }
        scale_grads(&mut acc, 0.5);
        sgd.step(&acc);
    }
    let got: Vec<f32> = params.iter().flat_map(|p| p.value()).collect();

    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-5, "accumulated {g} != full-batch {w}");
    }
}
