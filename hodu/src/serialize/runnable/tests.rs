use super::*;
use crate::Ctx;
use crate::kurumi::{CpuBackend, serialize_graph};
use crate::nn::{Embedding, Linear};

// A runnable artifact saved from a TRAINING Ctx must prune the backward nodes, then load and
// run from the file alone -- weights from the rows, the runtime input from the caller --
// recomputing the exact forward value.
#[test]
fn save_runnable_round_trips() {
    let ctx = Ctx::cpu();
    let lin = Linear::new(&ctx, 2, 1, 0);
    let x = ctx.input(vec![3, 2]);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ctx.feed(x.node(), xs.clone(), vec![3, 2]);
    let y = lin.forward(&x).unwrap();
    let want = ctx.eval_f32(y.node());

    // a training run: grad() grows the arena with backward nodes the artifact must drop.
    let params = lin.parameters();
    let pts: Vec<&Tensor> = params.iter().map(|p| p.tensor()).collect();
    let _ = y.grad(&pts).unwrap();

    let path = std::env::temp_dir().join("hodu_save_runnable_test.hodu");
    save_runnable(&path, &lin, &[&y], &[("x", &x)]).unwrap();

    // save_runnable prunes to the forward cone, so the blob is smaller than a whole-arena
    // serialize that would still carry the backward nodes.
    let (_, _, blob) = read_container(&path).unwrap();
    assert!(!blob.is_empty(), "the runnable artifact must carry a graph section");
    let mut whole: Vec<InputBinding> = lin
        .named_parameters("")
        .into_iter()
        .map(|(name, p)| InputBinding { node: p.tensor().node(), role: InputRole::Weight, name })
        .collect();
    whole.push(InputBinding { node: x.node(), role: InputRole::Runtime, name: "x".into() });
    let whole_blob = ctx.with_graph(|g| serialize_graph(g, &[y.node()], &whole));
    assert!(blob.len() < whole_blob.len(), "backward nodes not pruned ({} vs {})", blob.len(), whole_blob.len());

    // load + run through the public API: weights from the rows, "x" from the caller.
    let model = load_runnable(&path).unwrap();
    assert_eq!(model.input_names(), vec!["x"]);
    let got = model.run(&CpuBackend, &[("x", Storage::F32(xs))]).unwrap();
    assert_eq!(got[0].f32(), want.as_slice());

    std::fs::remove_file(&path).ok();
}

// A 2-entry artifact ("forward" + a scaled variant sharing the weights) saves, then the
// forward (entry 0) loads and runs from the file alone to the exact forward value.
#[test]
fn save_multi_forward_round_trips() {
    let ctx = Ctx::cpu();
    let lin = Linear::new(&ctx, 2, 1, 0);
    let x = ctx.input(vec![3, 2]);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    ctx.feed(x.node(), xs.clone(), vec![3, 2]);
    let y = lin.forward(&x).unwrap();
    let want = ctx.eval_f32(y.node());
    let y2 = y.relu(); // a second entry: a distinct output cone over the same shared weights

    let path = std::env::temp_dir().join("hodu_save_multi_test.hodu");
    save_multi(&path, &lin, &[("forward", &[&y], &[("x", &x)]), ("relu", &[&y2], &[("x", &x)])]).unwrap();

    // load_runnable loads entry 0 (forward) and runs it.
    let model = load_runnable(&path).unwrap();
    assert_eq!(model.input_names(), vec!["x"]);
    let got = model.run(&CpuBackend, &[("x", Storage::F32(xs))]).unwrap();
    assert_eq!(got[0].f32(), want.as_slice());

    std::fs::remove_file(&path).ok();
}

// The runtime input can be I64 tokens, not just f32: an Embedding forward is fed token ids
// through run's typed Storage, proving token/LM models deploy from the file. Before, run
// hardcoded Storage::F32 -> a dtype mismatch on the I64 index Input made this impossible.
#[test]
fn save_runnable_i64_tokens_round_trip() {
    let ctx = Ctx::cpu();
    let emb = Embedding::new(&ctx, 6, 4, 0);
    let idx = ctx.input_i64(vec![2, 3]);
    let ids: Vec<i64> = vec![0, 1, 2, 3, 4, 5];
    ctx.feed_i64(idx.node(), ids.clone(), vec![2, 3]);
    let y = emb.forward(&idx).unwrap();
    let want = ctx.eval_f32(y.node());

    let path = std::env::temp_dir().join("hodu_save_runnable_i64.hodu");
    save_runnable(&path, &emb, &[&y], &[("tokens", &idx)]).unwrap();

    let model = load_runnable(&path).unwrap();
    assert_eq!(model.input_names(), vec!["tokens"]);
    let got = model.run(&CpuBackend, &[("tokens", Storage::I64(ids))]).unwrap();
    assert_eq!(got[0].f32(), want.as_slice());

    std::fs::remove_file(&path).ok();
}

// Dev tool (run with `--ignored --nocapture`): (re)generate the checked-in cross-frontend
// fixture that hodu-py loads in tests/test_serialize.py, proving a Rust-written .hodu byte
// format + graph blob load and run in Python. Prints the input and expected output to
// hardcode there. Deterministic: Linear seed 0 + fixed x.
#[test]
#[ignore = "regenerates the committed hodu-py cross-frontend fixture; run manually"]
fn gen_cross_frontend_fixture() {
    let ctx = Ctx::cpu();
    let lin = Linear::new(&ctx, 4, 3, 0);
    let x = ctx.input(vec![2, 4]);
    let xs: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 - 0.4).collect();
    ctx.feed(x.node(), xs.clone(), vec![2, 4]);
    let y = lin.forward(&x).unwrap();
    let want = ctx.eval_f32(y.node());

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/linear.hodu");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    save_runnable(&path, &lin, &[&y], &[("x", &x)]).unwrap();

    println!("wrote {}", path.display());
    println!("x = {xs:?}");
    println!("want = {want:?}");
}

// Dev tool (run with `--ignored --nocapture`): (re)generate the checked-in cross-frontend
// DROPOUT runnable fixture hodu-py loads. A Linear -> Dropout net makes save_runnable bind
// the internal RNG Inputs (the train/eval flag + dropout seed) as reserved NUL-prefixed
// Weight bindings, so the artifact is self-contained -- load auto-feeds their eval-mode
// defaults. Saved in EVAL mode so the printed `want` equals the artifact's forced-eval
// forward (dropout = identity). Prints x + want to hardcode in the Python consumer.
#[test]
#[ignore = "regenerates the committed hodu-py cross-frontend dropout runnable fixture; run manually"]
fn gen_cross_frontend_dropout_fixture() {
    use crate::nn::{Dropout, Sequential};
    let ctx = Ctx::cpu();
    let model = Sequential::new(vec![
        Box::new(Linear::new(&ctx, 4, 3, 0)) as Box<dyn crate::nn::Module>,
        Box::new(Dropout::new(&ctx, 0.5).unwrap()),
    ]);
    let x = ctx.input(vec![2, 4]);
    let xs: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 - 0.4).collect();
    ctx.feed(x.node(), xs.clone(), vec![2, 4]);
    let y = model.forward(&x).unwrap();
    ctx.set_training(false); // eval: dropout is identity (flag 0.0), matching the artifact default
    let want = ctx.eval_f32(y.node());

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/dropout_runnable.hodu");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    save_runnable(&path, &model, &[&y], &[("x", &x)]).unwrap();

    println!("wrote {}", path.display());
    println!("x = {xs:?}");
    println!("want = {want:?}");
}

// Dev tool (run with --ignored --nocapture): (re)generate the cross-frontend K_BUFFER
// fixture. A plain save() of a BatchNorm carries its running stats as K_BUFFER rows; set
// them to distinctive values so hodu-py's row-level test pins the K_BUFFER byte codec by
// value (names diverge across frontends, so the Python side matches on the value set).
#[test]
#[ignore = "regenerates the committed hodu-py cross-frontend BatchNorm fixture; run manually"]
fn gen_cross_frontend_batchnorm_fixture() {
    let ctx = Ctx::cpu();
    let bn = crate::nn::BatchNorm2d::new(&ctx, 4, 1e-5, 0.1);
    let bufs = bn.named_buffers("");
    bufs[0].1.set(vec![1.0, 2.0, 3.0, 4.0]);
    bufs[1].1.set(vec![5.0, 6.0, 7.0, 8.0]);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/batchnorm.hodu");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    crate::serialize::save(&path, &bn).unwrap();

    println!("wrote {}", path.display());
    for (name, b) in bn.named_buffers("") {
        println!("K_BUFFER {name}: shape {:?} = {:?}", b.shape(), b.value());
    }
}

// Dev tool (run with --ignored --nocapture): (re)generate the cross-frontend QUANT fixture.
// A plain save() of a Sequential[QuantLinear] (asymmetric int4) writes the v2 quant-descriptor
// table; hodu-py reads it at the byte level to prove both frontends agree on the descriptor
// format (bits/group_size/symmetric + FQNs referencing the qweight/scales/mins rows).
#[test]
#[ignore = "regenerates the committed hodu-py cross-frontend quant fixture; run manually"]
fn gen_cross_frontend_quant_fixture() {
    use crate::nn::{QuantLinear, Sequential};
    let ctx = Ctx::cpu();
    let src = Linear::new(&ctx, 32, 8, 0); // in=32 (multiple of group_size=16), out=8
    let ql = QuantLinear::from_linear(&src, 4, 16, false).unwrap(); // int4, asymmetric -> mins present
    let model = Sequential::new(vec![Box::new(ql) as Box<dyn crate::nn::Module>]);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/quant.hodu");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    crate::serialize::save(&path, &model).unwrap();

    println!("wrote {}", path.display());
    for d in model.quant_descriptors("") {
        println!("descriptor {d:?}");
    }
}
