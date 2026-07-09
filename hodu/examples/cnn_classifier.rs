//! CNN image classification on the static engine. A synthetic 3-class texture set
//! (horizontal / vertical / diagonal bands + noise) is classified by
//! `Conv2d -> Relu -> MaxPool2d -> Flatten -> Linear`. The loss graph is built ONCE
//! at a fixed batch shape, then `DataLoader` feeds shuffled mini-batches to Adam.
//! Exercises conv2d + pooling + the data loader end-to-end (autodiff for free).
use hodu::prelude::*;

const IMG: usize = 12;
const CH: usize = 8;
const CLASSES: usize = 3;
const PER_CLASS: usize = 80;
const BATCH: usize = 40;
const EPOCHS: usize = 20;

fn main() {
    let ctx = Ctx::cpu();
    let (xs, labels) = make_images(PER_CLASS, 7);

    // conv -> pool -> flatten feature width, read from a dummy forward (12 --conv3p1--> 12
    // --pool2--> 6, so CH*6*6 = 288) instead of hand-counting.
    let feat_dim = feature_dim(&ctx);
    println!("conv/pool feature dim = {feat_dim}"); // 8*6*6 = 288

    let x = ctx.input(vec![BATCH, 1, IMG, IMG]);
    let targets = ctx.input(vec![BATCH, CLASSES]);
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(&ctx, 1, CH, (3, 3), (1, 1), (1, 1), 1)),
        Box::new(Relu),
        Box::new(MaxPool2d::new((2, 2), (2, 2))),
        Box::new(Flatten::new(1)),
        Box::new(Linear::new(&ctx, feat_dim, CLASSES, 2)),
    ]);
    let logits = model.forward(&x).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    let params = model.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap(); // built once
    let mut opt = Adam::new(params.clone(), 0.02);

    let mut train =
        DataLoader::new(Dataset::new(xs.clone(), vec![1, IMG, IMG], labels.clone()).unwrap(), BATCH, true, 42);
    for epoch in 0..EPOCHS {
        let (mut ep, mut nb) = (0.0f32, 0);
        for b in train.batches() {
            b.feed_x(&ctx, x.node());
            ctx.feed(targets.node(), one_hot(b.y_class(), CLASSES), vec![BATCH, CLASSES]);
            opt.step(&grad_values(&grads));
            ep += loss.item();
            nb += 1;
        }
        if epoch % 5 == 0 || epoch == EPOCHS - 1 {
            println!("epoch {epoch:>2}: loss {:.4}", ep / nb as f32);
        }
    }

    // accuracy over the (drop_last) dataset, batch by batch on the same graph.
    let mut eval = DataLoader::new(Dataset::new(xs, vec![1, IMG, IMG], labels).unwrap(), BATCH, false, 0);
    let (mut correct, mut total) = (0usize, 0usize);
    for b in eval.batches() {
        b.feed_x(&ctx, x.node());
        let lg = logits.realize();
        for (i, &lab) in b.y_class().iter().enumerate() {
            if argmax(&lg[i * CLASSES..(i + 1) * CLASSES]) == lab {
                correct += 1;
            }
            total += 1;
        }
    }
    println!("train accuracy {:.1}% ({correct}/{total})", 100.0 * correct as f32 / total as f32);
}

// flatten width of Conv2d->Relu->MaxPool2d->Flatten on a dummy image, via a
// throwaway ctx so no stray params land in the training graph.
fn feature_dim(_ctx: &Ctx) -> usize {
    let probe = Ctx::cpu();
    let x = probe.input(vec![1, 1, IMG, IMG]);
    probe.feed(x.node(), vec![0.0; IMG * IMG], vec![1, 1, IMG, IMG]);
    let w = Conv2d::new(&probe, 1, CH, (3, 3), (1, 1), (1, 1), 0);
    let feat = w.forward(&x).unwrap().relu().max_pool2d((2, 2), (2, 2)).unwrap().flatten(1).unwrap();
    feat.shape()[1]
}

// three separable textures + noise: horizontal / vertical / diagonal bands.
fn make_images(per_class: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut s = seed ^ 0x1234_5678;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for class in 0..CLASSES {
        for _ in 0..per_class {
            for r in 0..IMG {
                for c in 0..IMG {
                    let base = match class {
                        0 => ((r / 2) % 2) as f32,
                        1 => ((c / 2) % 2) as f32,
                        _ => (((r + c) / 2) % 2) as f32,
                    };
                    xs.push(base + noise(&mut s));
                }
            }
            ys.push(class);
        }
    }
    (xs, ys)
}

fn noise(s: &mut u64) -> f32 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32 * 0.3 - 0.15 // [-0.15, 0.15)
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}
