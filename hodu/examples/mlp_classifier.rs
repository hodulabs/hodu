//! 3-class blob classification on the static engine. The loss graph is built ONCE
//! (`cross_entropy(Sequential(x), one_hot)` over Input leaves), then Adam trains it
//! by feeding the full batch each step. Exercises Sequential, activations, softmax
//! cross-entropy, autograd, and Adam.
use hodu::prelude::*;

fn main() {
    let ctx = Ctx::cpu();
    let (xs, labels) = blobs();
    let n = labels.len();

    let x = ctx.input(vec![n, 2]);
    ctx.feed(x.node(), xs, vec![n, 2]);
    let targets = ctx.input(vec![n, 3]);
    ctx.feed(targets.node(), one_hot(&labels, 3), vec![n, 3]);

    let model = Sequential::new(vec![
        Box::new(Linear::new(&ctx, 2, 32, 1)),
        Box::new(Relu),
        Box::new(Linear::new(&ctx, 32, 3, 2)),
    ]);
    let logits = model.forward(&x).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    let params = model.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap(); // built once
    let mut opt = Adam::new(params.clone(), 0.05);

    for step in 0..300 {
        opt.step(&grad_values(&grads));
        if step % 50 == 0 {
            println!("step {step:>3}: loss {:.4}", loss.item());
        }
    }
    println!("final loss {:.4}", loss.item());
    println!("train accuracy {:.1}%", 100.0 * accuracy(&logits, &labels, 3));
}

// three well-separated 2-D blobs (uniform jitter), 100 points each.
fn blobs() -> (Vec<f32>, Vec<usize>) {
    let centers = [(2.5f32, 2.5f32), (-2.5, 2.5), (0.0, -2.5)];
    let mut s = 12345u64;
    let mut jitter = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 1.6 - 0.8 // [-0.8, 0.8)
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..100 {
            xs.push(cx + jitter());
            xs.push(cy + jitter());
            ys.push(c);
        }
    }
    (xs, ys)
}

fn one_hot(labels: &[usize], classes: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; labels.len() * classes];
    for (i, &c) in labels.iter().enumerate() {
        o[i * classes + c] = 1.0;
    }
    o
}
