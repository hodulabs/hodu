//! Generative causal LM end-to-end: Embedding -> causal + RoPE Transformer -> Linear to
//! vocab, trained with cross_entropy over shifted targets. The task is "predict the
//! PREVIOUS token": the output at position t must be the token at t-1 (position 0 -> the
//! BOS id 0). Solving it means routing token[t-1]'s identity into position t's output,
//! which needs the causal mask (position t may attend to t-1) plus RoPE's relative
//! position (learn "attend one step back"). First exercise of the `causal=true` attention
//! path from the frontend -- all other transformer coverage is causal=false.
use hodu::prelude::*;

const V: usize = 8; // vocab; id 0 reserved as the position-0 (BOS) target
const S: usize = 6; // sequence length
const B: usize = 16; // sequences in the single fixed batch
const D: usize = 32;
const HEADS: usize = 4;
const LAYERS: usize = 1; // one causal+RoPE block already solves copy-the-previous

#[test]
fn causal_lm_learns_previous_token() {
    let ctx = Ctx::cpu();
    let (tokens, target_ids) = make_data(0xBADC0DE);

    let idx = ctx.input_i64(vec![B, S]);
    ctx.feed_i64(idx.node(), tokens, vec![B, S]);
    let targets = ctx.input(vec![B * S, V]);
    ctx.feed(targets.node(), one_hot(&target_ids, V), vec![B * S, V]);

    let emb = Embedding::new(&ctx, V, D, 1);
    let enc = TransformerEncoder::new(&ctx, D, HEADS, LAYERS, true, true, 2).unwrap(); // causal + RoPE
    let head = Linear::new(&ctx, D, V, 3);

    let logits = forward(&emb, &enc, &head, &idx);
    let loss = cross_entropy(&logits, &targets).unwrap();

    let mut params = emb.parameters();
    params.extend(enc.parameters());
    params.extend(head.parameters());
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.02);

    let l0 = loss.item();
    for _ in 0..150 {
        opt.step(&grad_values(&grads));
    }
    let l1 = loss.item();
    let acc = accuracy(&logits, &target_ids, V);
    eprintln!("causal LM: loss {l0:.4} -> {l1:.4}, acc {:.1}% (chance {:.1}%)", acc * 100.0, 100.0 / V as f32);
    assert!(l1 < l0, "loss did not drop: {l0} -> {l1}");
    // chance = 1/V = 12.5%; a working causal+RoPE path copies the previous token far above it.
    assert!(acc > 0.6, "causal LM accuracy not clearly above chance: {acc} (loss {l0} -> {l1})");
}

// idx [B,S] -> emb [B,S,D] -> encoder [B,S,D] -> flatten [B*S,D] -> vocab logits [B*S,V].
fn forward(emb: &Embedding, enc: &TransformerEncoder, head: &Linear, idx: &Tensor) -> Tensor {
    let h = emb.forward(idx).unwrap();
    let h = enc.forward(&h).unwrap();
    let flat = h.reshape(vec![B * S, D]).unwrap();
    head.forward(&flat).unwrap()
}

// Random tokens in [1, V-1]; target[t] = token[t-1], target[0] = 0 (BOS). Flat [B*S].
fn make_data(seed: u64) -> (Vec<i64>, Vec<usize>) {
    let mut s = seed ^ 0xDEAD_BEEF;
    let mut tokens = Vec::with_capacity(B * S);
    let mut targets = Vec::with_capacity(B * S);
    for _ in 0..B {
        let seq: Vec<i64> = (0..S).map(|_| 1 + (next(&mut s) % (V as u64 - 1)) as i64).collect();
        for t in 0..S {
            tokens.push(seq[t]);
            targets.push(if t == 0 { 0 } else { seq[t - 1] as usize });
        }
    }
    (tokens, targets)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
