//! End-to-end Transformer: token Embedding -> RoPE MultiHeadAttention blocks ->
//! mean-pool -> Linear learns a synthetic sequence-classification task that
//! genuinely needs cross-position mixing, trained on kurumi's build-once/feed graph.
//!
//! Task ("marker after the trigger"): each length-8 sequence over a 12-token vocab
//! contains exactly one trigger token (id 0) at a random position `p`; the token at
//! `p+1` is one of three marker tokens {1,2,3} and *is* the label. A second, always
//! different marker sits at some other position as a distractor, and the rest are
//! noise tokens {4..11}. Because two distinct markers are present, a bag-of-tokens
//! (embedding + pool with no attention) cannot decide which is the answer -- only
//! "the marker immediately after the trigger" resolves it, which needs relative
//! position (RoPE) + attention. Bag-of-tokens tops out near 50%; the model reaches
//! near 100%.
use hodu::prelude::*;

const VOCAB: usize = 12;
const SEQ: usize = 8;
const D: usize = 16;
const HEADS: usize = 4;
const CLASSES: usize = 3;
const LAYERS: usize = 2;
const BATCH: usize = 32;
const PER_CLASS: usize = 32; // 96 samples -> 3 full batches
const EPOCHS: usize = 10;

#[test]
fn transformer_classifies_marker_after_trigger() {
    let ctx = Ctx::cpu();
    let (seqs, labels) = make_data(PER_CLASS, 0xC0FFEE);
    let n = labels.len();

    // Build the loss graph once over a fixed [BATCH, SEQ] token slot + one-hot slot.
    let idx = ctx.input_i64(vec![BATCH, SEQ]);
    let targets = ctx.input(vec![BATCH, CLASSES]);

    let emb = Embedding::new(&ctx, VOCAB, D, 1);
    let enc = TransformerEncoder::new(&ctx, D, HEADS, LAYERS, false, true, 2).unwrap();
    let head = Linear::new(&ctx, D, CLASSES, 3);

    let logits = forward(&emb, &enc, &head, &idx);
    let loss = cross_entropy(&logits, &targets).unwrap();

    let mut params = emb.parameters();
    params.extend(enc.parameters());
    params.extend(head.parameters());
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    // `grad_values` realizes all ~35 grads in one shared pass, so the forward+backward
    // trunk common to every grad computes once (no per-param re-traversal cliff).
    let mut opt = Adam::new(params.clone(), 0.01);

    // A fixed reference batch (mixed classes) so l0 and l1 measure the same thing.
    let refb: Vec<usize> = (0..BATCH).map(|i| (i * (n / BATCH)) % n).collect();
    feed_batch(&ctx, &idx, &targets, &seqs, &labels, &refb);
    let l0 = loss.item();

    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = 777u64;
    for _ in 0..EPOCHS {
        for i in (1..n).rev() {
            order.swap(i, (next(&mut rng) as usize) % (i + 1));
        }
        for chunk in order.chunks_exact(BATCH) {
            feed_batch(&ctx, &idx, &targets, &seqs, &labels, chunk);
            opt.step(&grad_values(&grads));
        }
    }

    feed_batch(&ctx, &idx, &targets, &seqs, &labels, &refb);
    let l1 = loss.item();
    let acc = train_accuracy(&ctx, &idx, &targets, &logits, &seqs, &labels);
    eprintln!("transformer seqcls: loss {l0:.4} -> {l1:.4}, train acc {:.1}%", acc * 100.0);
    assert!(l1 < l0, "loss did not drop: {l0} -> {l1}");
    assert!(acc > 0.9, "train accuracy too low: {acc} (loss {l0} -> {l1})");
}

// A d_model not divisible by n_heads must Err at construction (not panic), and the
// error propagates out of the block/encoder that build a MultiHeadAttention.
#[test]
fn indivisible_heads_error() {
    let ctx = Ctx::cpu();
    assert!(
        MultiHeadAttention::new(&ctx, 10, 4, false, false, 0).is_err(),
        "d_model 10 not divisible by 4 heads must Err"
    );
    assert!(
        TransformerBlock::new(&ctx, 10, 4, false, false, 0).is_err(),
        "block must propagate the attention build error"
    );
    assert!(
        TransformerEncoder::new(&ctx, 10, 4, 2, false, false, 0).is_err(),
        "encoder must propagate the attention build error"
    );
    // divisible -> Ok.
    assert!(TransformerEncoder::new(&ctx, 12, 4, 2, false, false, 0).is_ok());
}

// A wrong-rank input Errs at the attention layer instead of panicking on the unchecked
// [B,S,d] shape index -- it wants rank 3.
#[test]
fn attention_wrong_rank_errors() {
    let ctx = Ctx::cpu();
    let mha = MultiHeadAttention::new(&ctx, D, HEADS, false, false, 0).unwrap();
    assert!(mha.forward(&ctx.zeros(vec![BATCH, D])).is_err()); // rank 2, missing the seq axis
}

// emb -> encoder -> mean-pool over sequence -> classifier.
fn forward(emb: &Embedding, enc: &TransformerEncoder, head: &Linear, idx: &Tensor) -> Tensor {
    let h = emb.forward(idx).unwrap(); // [B, S, D]
    let h = enc.forward(&h).unwrap(); // [B, S, D]
    let pooled = h.mean_axis(1).unwrap(); // [B, D]
    head.forward(&pooled).unwrap() // [B, CLASSES]
}

fn feed_batch(ctx: &Ctx, idx: &Tensor, targets: &Tensor, seqs: &[i64], labels: &[usize], chunk: &[usize]) {
    let mut bx: Vec<i64> = Vec::with_capacity(BATCH * SEQ);
    let mut by: Vec<usize> = Vec::with_capacity(BATCH);
    for &s in chunk {
        bx.extend_from_slice(&seqs[s * SEQ..(s + 1) * SEQ]);
        by.push(labels[s]);
    }
    ctx.feed_i64(idx.node(), bx, vec![BATCH, SEQ]);
    ctx.feed(targets.node(), one_hot(&by, CLASSES), vec![BATCH, CLASSES]);
}

fn train_accuracy(ctx: &Ctx, idx: &Tensor, targets: &Tensor, logits: &Tensor, seqs: &[i64], labels: &[usize]) -> f32 {
    let (mut correct, mut total) = (0usize, 0usize);
    for chunk in (0..labels.len()).collect::<Vec<_>>().chunks_exact(BATCH) {
        feed_batch(ctx, idx, targets, seqs, labels, chunk);
        let preds = argmax(logits, 1);
        for (i, &s) in chunk.iter().enumerate() {
            if preds[i] == labels[s] {
                correct += 1;
            }
            total += 1;
        }
    }
    correct as f32 / total as f32
}

// Sequence generator: exactly one trigger (0), the answer marker right after it, a
// different marker as a distractor elsewhere, noise (4..11) in the rest.
fn make_data(per_class: usize, seed: u64) -> (Vec<i64>, Vec<usize>) {
    let mut s = seed ^ 0xDEAD_BEEF;
    let mut seqs: Vec<i64> = Vec::with_capacity(per_class * CLASSES * SEQ);
    let mut labels: Vec<usize> = Vec::with_capacity(per_class * CLASSES);
    for class in 0..CLASSES {
        for _ in 0..per_class {
            let marker_true = 1 + class as i64;
            let p = (next(&mut s) as usize) % (SEQ - 1); // trigger position, leaves p+1
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

fn one_hot(labels: &[usize], classes: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; labels.len() * classes];
    for (i, &c) in labels.iter().enumerate() {
        o[i * classes + c] = 1.0;
    }
    o
}

// splitmix64 draw.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
