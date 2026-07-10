use super::*;
use crate::Ctx;
use crate::kurumi::{CpuBackend, serialize_graph};
use crate::nn::Linear;

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
    let (_, blob) = read_container(&path).unwrap();
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
    let got = model.run(&CpuBackend, &[("x", &xs)]).unwrap();
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
