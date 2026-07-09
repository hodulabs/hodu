//! Train a small RoPE Transformer to classify synthetic token sequences.
//!
//! Task ("marker after the trigger"): each length-8 sequence over a 12-token vocab
//! has one trigger token (id 0) at a random position `p`; the token at `p+1` is one
//! of three markers {1,2,3} and *is* the label. A second, different marker sits
//! elsewhere as a distractor, and the rest are noise {4..11}. Two distinct markers
//! are present, so an embedding + pool with no attention cannot tell which is the
//! answer -- only "the marker right after the trigger" resolves it, which needs
//! relative position (RoPE) + attention. Model: Embedding -> 2 pre-norm
//! TransformerBlocks (4 heads, RoPE) -> mean-pool -> Linear, softmax cross-entropy,
//! trained with Adam. Token batches come from the generalized `DataLoader` (i64).
//!
//! Run: `cargo run --example transformer_seqcls`
use hodu::prelude::*;

const VOCAB: usize = 12;
const SEQ: usize = 8;
const D: usize = 16;
const HEADS: usize = 4;
const CLASSES: usize = 3;
const LAYERS: usize = 2;
const BATCH: usize = 32;
const PER_CLASS: usize = 32;
const EPOCHS: usize = 15;

fn main() {
    let ctx = Ctx::cpu();
    let (seqs, labels) = make_data(PER_CLASS, 0xC0FFEE);

    let idx = ctx.input_i64(vec![BATCH, SEQ]);
    let targets = ctx.input(vec![BATCH, CLASSES]);

    let emb = Embedding::new(&ctx, VOCAB, D, 1);
    let enc = TransformerEncoder::new(&ctx, D, HEADS, LAYERS, false, true, 2).unwrap();
    let head = Linear::new(&ctx, D, CLASSES, 3);

    // Build the loss graph once; feed integer token batches + one-hot labels per step.
    let h = emb.forward(&idx).unwrap();
    let h = enc.forward(&h).unwrap();
    let pooled = h.mean_axis(1).unwrap();
    let logits = head.forward(&pooled).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();

    let mut params = emb.parameters();
    params.extend(enc.parameters());
    params.extend(head.parameters());
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.01);
    println!("params: {} tensors", params.len());

    // i64 token sequences flow through the generalized DataLoader (shuffle per epoch).
    let mut dl = DataLoader::new(Dataset::tokens(seqs.clone(), vec![SEQ], labels.clone()).unwrap(), BATCH, true, 777);
    for epoch in 0..EPOCHS {
        let (mut ep, mut steps) = (0.0f32, 0);
        for b in dl.batches() {
            b.feed_x(&ctx, idx.node());
            ctx.feed(targets.node(), one_hot(b.y_class(), CLASSES), vec![BATCH, CLASSES]);
            opt.step(&grad_values(&grads));
            ep += loss.item();
            steps += 1;
        }
        if epoch % 5 == 0 || epoch + 1 == EPOCHS {
            println!("epoch {epoch:2}: loss {:.4}", ep / steps as f32);
        }
    }

    let mut eval = DataLoader::new(Dataset::tokens(seqs, vec![SEQ], labels).unwrap(), BATCH, false, 0);
    let (mut correct, mut total) = (0usize, 0usize);
    for b in eval.batches() {
        b.feed_x(&ctx, idx.node());
        let lg = logits.realize();
        for (i, &lab) in b.y_class().iter().enumerate() {
            if argmax(&lg[i * CLASSES..(i + 1) * CLASSES]) == lab {
                correct += 1;
            }
            total += 1;
        }
    }
    println!("final train accuracy: {:.1}%", correct as f32 / total as f32 * 100.0);
}

fn make_data(per_class: usize, seed: u64) -> (Vec<i64>, Vec<usize>) {
    let mut s = seed ^ 0xDEAD_BEEF;
    let mut seqs: Vec<i64> = Vec::with_capacity(per_class * CLASSES * SEQ);
    let mut labels: Vec<usize> = Vec::with_capacity(per_class * CLASSES);
    for class in 0..CLASSES {
        for _ in 0..per_class {
            let marker_true = 1 + class as i64;
            let p = (next(&mut s) as usize) % (SEQ - 1);
            let others: Vec<i64> = (1..=3i64).filter(|&m| m != marker_true).collect();
            let distractor = others[(next(&mut s) as usize) % others.len()];
            let cands: Vec<usize> = (0..SEQ).filter(|&i| i != p && i != p + 1).collect();
            let q = cands[(next(&mut s) as usize) % cands.len()];
            let mut seq: Vec<i64> = (0..SEQ).map(|_| 4 + (next(&mut s) % 8) as i64).collect();
            seq[p] = 0;
            seq[p + 1] = marker_true;
            seq[q] = distractor;
            seqs.extend_from_slice(&seq);
            labels.push(class);
        }
    }
    (seqs, labels)
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
