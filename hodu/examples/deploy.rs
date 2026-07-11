//! The deploy path: train a tiny MLP, save it as a self-contained runnable `.hodu`, reload
//! from the file alone, and check the artifact reproduces the in-process forward. The trained
//! graph + its weights ship as one file that `load_runnable(...).run(...)` executes.
//!
//! Run: `cargo run --example deploy`
use hodu::kurumi::{CpuBackend, Storage};
use hodu::prelude::*;

fn main() {
    let ctx = Ctx::cpu();

    // Toy 2-D, 3-class data (three well-separated blobs), the whole set as one batch.
    let (xs, labels) = blobs();
    let n = labels.len();
    let x = ctx.input(vec![n, 2]);
    ctx.feed(x.node(), xs.clone(), vec![n, 2]);
    let targets = ctx.input(vec![n, 3]);
    ctx.feed(targets.node(), one_hot(&labels, 3), vec![n, 3]);

    // A 2 -> 16 -> 3 classifier.
    let model = Sequential::new(vec![
        Box::new(Linear::new(&ctx, 2, 16, 1)),
        Box::new(Relu),
        Box::new(Linear::new(&ctx, 16, 3, 2)),
    ]);
    let logits = model.forward(&x).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    // Train a few hundred steps of Adam over the fixed batch.
    let params = model.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.05);
    for _ in 0..300 {
        opt.step(&grad_values(&grads));
    }
    println!("trained: loss {:.4}, accuracy {:.1}%", loss.item(), 100.0 * accuracy(&logits, &labels, 3));

    // The in-process forward the artifact must reproduce exactly.
    let want = logits.realize();

    // Save the forward graph + weights as a self-contained runnable artifact, then drop
    // the training ctx entirely -- everything the model needs now lives in the file.
    let path = std::env::temp_dir().join("hodu_deploy_mlp.hodu");
    save_runnable(&path, &model, &[&logits], &[("x", &x)]).unwrap();
    println!("saved runnable to {}", path.display());

    // Reload and run from the file alone: weights resolved from the rows, "x" fed by us.
    let runnable = load_runnable(&path).unwrap();
    println!("artifact runtime inputs: {:?}", runnable.input_names());
    let got = runnable.run(&CpuBackend, &[("x", Storage::F32(xs))]).unwrap();
    let got = got[0].f32();

    let max_err = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("reload matches in-process forward: max abs err {max_err:.2e}");
    assert!(max_err < 1e-5, "artifact forward diverged from training forward: {max_err}");
    std::fs::remove_file(&path).ok();
    println!("deploy OK");
}

// Three well-separated 2-D blobs (uniform jitter), 40 points each.
fn blobs() -> (Vec<f32>, Vec<usize>) {
    let centers = [(2.5f32, 2.5f32), (-2.5, 2.5), (0.0, -2.5)];
    let mut s = 12345u64;
    let mut jitter = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 1.6 - 0.8
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..40 {
            xs.push(cx + jitter());
            xs.push(cy + jitter());
            ys.push(c);
        }
    }
    (xs, ys)
}
