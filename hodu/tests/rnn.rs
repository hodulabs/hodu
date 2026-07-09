//! End-to-end proof that the unrolled recurrent layers learn a task that genuinely
//! needs memory of ORDER. Task ("which marker comes first"): each length-T one-hot
//! sequence contains exactly one A, one B, and T-2 fillers; the label is whether A
//! appears before B. The token MULTISET is identical across the two classes (one A,
//! one B, T-2 fillers), so a bag-of-tokens / mean-pool model sees the same features
//! for both labels and is stuck at 50%. Only a model that tracks order -- the cell
//! latching which marker it saw first and carrying it to the last hidden -- solves
//! it. We assert final train accuracy > 90% for both LSTM and GRU.
use hodu::prelude::*;

const T: usize = 10; // sequence length (baked into the unrolled graph)
const IN: usize = 3; // one-hot channels: A=0, B=1, filler=2
const H: usize = 32;
const CLASSES: usize = 2;
const BATCH: usize = 32;

#[test]
fn lstm_learns_marker_order() {
    let ctx = Ctx::cpu();
    let lstm = Lstm::new(&ctx, IN, H, 1);
    let head = Linear::new(&ctx, H, CLASSES, 2);
    let mut params = lstm.parameters();
    params.extend(head.parameters());
    let acc = train(&ctx, 192, 25, |xin| head.forward(&lstm.forward(xin)?), params);
    assert!(acc > 0.90, "LSTM train accuracy too low: {acc}");
}

#[test]
fn gru_learns_marker_order() {
    let ctx = Ctx::cpu();
    let gru = Gru::new(&ctx, IN, H, 1);
    let head = Linear::new(&ctx, H, CLASSES, 2);
    let mut params = gru.parameters();
    params.extend(head.parameters());
    let acc = train(&ctx, 160, 25, |xin| head.forward(&gru.forward(xin)?), params);
    assert!(acc > 0.90, "GRU train accuracy too low: {acc}");
}

#[test]
fn return_sequences_stacks_time() {
    // `return_sequences` re-stacks per-timestep hiddens into `[B,T,H]`.
    let ctx = Ctx::cpu();
    let x = ctx.zeros(vec![2, 4, IN]);
    let lstm = Lstm::new(&ctx, IN, H, 1).return_sequences();
    let y = lstm.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 4, H]);
    // zero input + zero init state => zero gates => all hiddens stay zero.
    assert!(y.realize().iter().all(|&v| v == 0.0));
}

// Build the loss graph once, train with Adam over `epochs`, return final train acc.
fn train(
    ctx: &Ctx,
    per_class: usize,
    epochs: usize,
    model: impl Fn(&Tensor) -> Result<Tensor, Error>,
    params: Vec<Param>,
) -> f32 {
    let (x, y) = make_data(per_class, 0xA11CE);
    let ds = Dataset::new(x, vec![T, IN], y).unwrap();
    let mut dl = DataLoader::new(ds, BATCH, true, 7);

    let xin = ctx.input(vec![BATCH, T, IN]);
    let targets = ctx.input(vec![BATCH, CLASSES]);
    let logits = model(&xin).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.01);

    let mut l0 = None;
    for _ in 0..epochs {
        for b in dl.batches() {
            b.feed_x(ctx, xin.node());
            ctx.feed(targets.node(), one_hot(b.y_class(), CLASSES), vec![BATCH, CLASSES]);
            if l0.is_none() {
                l0 = Some(loss.item());
            }
            opt.step(&grad_values(&grads));
        }
    }

    let (mut correct, mut total, mut lf) = (0usize, 0usize, 0.0);
    for b in dl.batches() {
        b.feed_x(ctx, xin.node());
        ctx.feed(targets.node(), one_hot(b.y_class(), CLASSES), vec![BATCH, CLASSES]);
        lf = loss.item();
        let lg = logits.realize();
        for (i, &lab) in b.y_class().iter().enumerate() {
            if argmax(&lg[i * CLASSES..(i + 1) * CLASSES]) == lab {
                correct += 1;
            }
            total += 1;
        }
    }
    let acc = correct as f32 / total as f32;
    println!("loss {:.4} -> {lf:.4}, train acc {:.1}%", l0.unwrap(), acc * 100.0);
    acc
}

// One-hot [N*T*IN] over the "which marker first" task; label = class (0 = A first).
fn make_data(per_class: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut s = seed ^ 0xDEAD_BEEF;
    let mut x = vec![0.0f32; per_class * CLASSES * T * IN];
    let mut y = Vec::with_capacity(per_class * CLASSES);
    let mut w = 0usize;
    for class in 0..CLASSES {
        for _ in 0..per_class {
            // two distinct positions i < j; the earlier one holds the "first" marker.
            let i = (next(&mut s) as usize) % (T - 1);
            let j = i + 1 + (next(&mut s) as usize) % (T - 1 - i);
            let (first, second) = if class == 0 { (0, 1) } else { (1, 0) }; // A=0, B=1
            for p in 0..T {
                let sym = if p == i {
                    first
                } else if p == j {
                    second
                } else {
                    2 // filler
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
