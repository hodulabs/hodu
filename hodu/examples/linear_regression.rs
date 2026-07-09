//! Linear regression on `y = 3x - 2`, trained end-to-end on the static engine:
//! the loss graph is built ONCE (x, y, params as Inputs), then each step feeds
//! the batch, realizes the grads, and SGD updates the params. Exercises the whole
//! stack -- broadcasting, operators, matmul, autograd, feeds, optim.
use hodu::prelude::*;

fn main() {
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
    let grads = loss.grad(&pts).unwrap(); // built once
    let sgd = Sgd::new(params.clone(), 0.2);

    for step in 0..300 {
        sgd.step(&grad_values(&grads));
        if step % 50 == 0 {
            println!("step {step:>3}: loss {:.6}", loss.item());
        }
    }

    println!("final loss {:.6}", loss.item());
    println!("recovered  w = {:.4}, b = {:.4}", params[0].value()[0], params[1].value()[0]);
    println!("target     w = 3.0000, b = -2.0000");
}
