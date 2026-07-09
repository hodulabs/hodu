//! Train an unrolled LSTM on a task that genuinely needs recurrent memory of ORDER,
//! and show a bag-of-tokens baseline fail on the SAME data.
//!
//! Task ("which marker comes first"): each length-T one-hot sequence holds exactly
//! one A, one B, and T-2 fillers; the label is whether A comes before B. The token
//! MULTISET is identical for both classes (one A, one B, T-2 fillers), so any model
//! that only sees token counts -- a mean-pool / bag-of-tokens -- gets identical
//! features for both labels and cannot beat 50%. Only order resolves it: the LSTM
//! latches which marker it saw first and carries that to its last hidden state.
//!
//! Model: `Lstm(IN, H)` -> last hidden `[B,H]` -> `Linear(H, 2)`, softmax
//! cross-entropy, Adam, on kurumi's build-once / feed-per-step static graph.
//!
//! The graph is static => the unroll length T is baked in at build time, so every
//! fed batch must have the same T.
//!
//! Run: `cargo run --release --example lstm_seqcls`
use hodu::prelude::*;

const T: usize = 10;
const IN: usize = 3; // one-hot: A=0, B=1, filler=2
const H: usize = 32;
const CLASSES: usize = 2;
const PER_CLASS: usize = 256;
const BATCH: usize = 32;
const EPOCHS: usize = 40;

fn main() {
    let ctx = Ctx::cpu();
    let (x, y) = make_data(PER_CLASS, 0xA11CE);

    // --- LSTM: reads order, should solve the task ---
    let lstm = Lstm::new(&ctx, IN, H, 1);
    let head = Linear::new(&ctx, H, CLASSES, 2);
    let lstm_acc = train(
        &ctx,
        &x,
        &y,
        |xin| head.forward(&lstm.forward(xin)?),
        {
            let mut p = lstm.parameters();
            p.extend(head.parameters());
            p
        },
        "LSTM (recurrent)",
    );

    // --- Bag-of-tokens baseline: mean over time then an MLP; blind to order ---
    let l1 = Linear::new(&ctx, IN, 16, 3);
    let l2 = Linear::new(&ctx, 16, CLASSES, 4);
    let bag_acc = train(
        &ctx,
        &x,
        &y,
        |xin| l2.forward(&l1.forward(&xin.mean_axis(1)?)?.relu()),
        {
            let mut p = l1.parameters();
            p.extend(l2.parameters());
            p
        },
        "bag-of-tokens (mean-pool + MLP)",
    );

    println!("\nLSTM train acc {:.1}%  vs  bag-of-tokens {:.1}% (chance 50%)", lstm_acc * 100.0, bag_acc * 100.0);
    println!("-> the multiset is identical across classes, so only the order-aware LSTM can win.");
}

// Build the loss graph once from `model`, train with Adam, return final train acc.
fn train(
    ctx: &Ctx,
    x: &[f32],
    y: &[usize],
    model: impl Fn(&Tensor) -> Result<Tensor, Error>,
    params: Vec<Param>,
    name: &str,
) -> f32 {
    let xin = ctx.input(vec![BATCH, T, IN]);
    let targets = ctx.input(vec![BATCH, CLASSES]);
    let logits = model(&xin).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.01);

    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = 777u64;
    println!("\n[{name}] {} params", params.len());
    for epoch in 0..EPOCHS {
        for i in (1..n).rev() {
            order.swap(i, (next(&mut rng) as usize) % (i + 1));
        }
        let (mut el, mut steps) = (0.0, 0);
        for chunk in order.chunks_exact(BATCH) {
            feed(ctx, &xin, &targets, x, y, chunk);
            opt.step(&grad_values(&grads));
            el += loss.item();
            steps += 1;
        }
        if epoch % 10 == 0 || epoch + 1 == EPOCHS {
            println!("  epoch {epoch:2}: loss {:.4}", el / steps as f32);
        }
    }

    let (mut correct, mut total) = (0usize, 0usize);
    for chunk in (0..n).collect::<Vec<_>>().chunks_exact(BATCH) {
        feed(ctx, &xin, &targets, x, y, chunk);
        let lg = logits.realize();
        for (i, &s) in chunk.iter().enumerate() {
            if argmax(&lg[i * CLASSES..(i + 1) * CLASSES]) == y[s] {
                correct += 1;
            }
            total += 1;
        }
    }
    correct as f32 / total as f32
}

fn feed(ctx: &Ctx, xin: &Tensor, targets: &Tensor, x: &[f32], y: &[usize], chunk: &[usize]) {
    let mut bx = Vec::with_capacity(BATCH * T * IN);
    let mut by = Vec::with_capacity(BATCH);
    for &s in chunk {
        bx.extend_from_slice(&x[s * T * IN..(s + 1) * T * IN]);
        by.push(y[s]);
    }
    ctx.feed(xin.node(), bx, vec![BATCH, T, IN]);
    ctx.feed(targets.node(), one_hot(&by, CLASSES), vec![BATCH, CLASSES]);
}

fn make_data(per_class: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut s = seed ^ 0xDEAD_BEEF;
    let mut x = vec![0.0f32; per_class * CLASSES * T * IN];
    let mut y = Vec::with_capacity(per_class * CLASSES);
    let mut w = 0usize;
    for class in 0..CLASSES {
        for _ in 0..per_class {
            let i = (next(&mut s) as usize) % (T - 1);
            let j = i + 1 + (next(&mut s) as usize) % (T - 1 - i);
            let (first, second) = if class == 0 { (0, 1) } else { (1, 0) };
            for p in 0..T {
                let sym = if p == i {
                    first
                } else if p == j {
                    second
                } else {
                    2
                };
                x[w * IN + sym] = 1.0;
                w += 1;
            }
            y.push(class);
        }
    }
    (x, y)
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, _)| k).unwrap_or(0)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
