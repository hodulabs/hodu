//! Metal backend seam: the same static graph trains on Metal via `Ctx::metal()`.
//! Skips cleanly on a machine with no Metal device. Correctness is the engine's
//! contract (unsupported ops fall back to the CPU oracle), so Metal must reach
//! the same optimum as CPU.
use hodu::prelude::*;

// y = 3x - 2 fit, run on the Metal backend; assert it converges like CPU.
#[test]
fn metal_trains_linear_regression() {
    let Some(ctx) = Ctx::metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
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

    let final_loss = loss.item();
    assert!(final_loss < 1e-3, "metal loss {final_loss} not converged");
    assert!((params[0].value()[0] - 3.0).abs() < 0.05);
    assert!((params[1].value()[0] + 2.0).abs() < 0.05);
}
