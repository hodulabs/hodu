//! safetensors interop: apply a `.safetensors` file into a live model by FQN reproduces the
//! source weights, name_map remaps HF-style names, strictness rejects missing/extra/mismatch,
//! and the cross-frontend fixture written by hodu-py (tests/fixtures/weights.safetensors) reads
//! back with the same values -- proof both frontends agree on the public standard.
use hodu::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::path::Path;

fn mlp(ctx: &Ctx, seed: u64) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(ctx, 3, 4, seed)),
        Box::new(Relu),
        Box::new(Linear::new(ctx, 4, 2, seed + 1)),
    ])
}

// Write f32 tensors as a real .safetensors via the crate (the export path).
fn write_f32(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let bufs: Vec<Vec<u8>> = tensors.iter().map(|(_, _, v)| v.iter().flat_map(|x| x.to_le_bytes()).collect()).collect();
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .zip(&bufs)
        .map(|((n, shape, _), b)| (n.clone(), TensorView::new(Dtype::F32, shape.clone(), b).unwrap()))
        .collect();
    std::fs::write(path, safetensors::serialize(views, None).unwrap()).unwrap();
}

// A model's params as (fqn, shape, values), so a fresh model can be warm-started from them.
fn model_f32(model: &dyn Module) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    model.named_parameters("").into_iter().map(|(n, p)| (n, p.shape().to_vec(), p.value())).collect()
}

#[test]
fn apply_reproduces_weights() {
    let path = std::env::temp_dir().join("hodu_st_apply.safetensors");
    let ctx = Ctx::cpu();
    let trained = mlp(&ctx, 0);
    let x = ctx.constant(vec![0.5, -1.0, 2.0], vec![1, 3]);
    let want = trained.forward(&x).unwrap().realize();
    write_f32(&path, &model_f32(&trained));

    // a fresh model with a DIFFERENT init -> different output until warm-started.
    let ctx2 = Ctx::cpu();
    let fresh = mlp(&ctx2, 100);
    let x2 = ctx2.constant(vec![0.5, -1.0, 2.0], vec![1, 3]);
    assert_ne!(fresh.forward(&x2).unwrap().realize(), want, "different init should differ before apply");

    apply_safetensors(&fresh, &path, |s| s.to_string()).unwrap();
    assert_eq!(fresh.forward(&x2).unwrap().realize(), want, "applied weights must reproduce the source model");
    std::fs::remove_file(&path).ok();
}

#[test]
fn name_map_remaps_hf_names() {
    let path = std::env::temp_dir().join("hodu_st_namemap.safetensors");
    let ctx = Ctx::cpu();
    let src = Linear::new(&ctx, 3, 2, 7);
    // store under HF-ish names; the model's FQNs are "0" (weight) and "1" (bias).
    let named: Vec<(String, Vec<usize>, Vec<f32>)> = src
        .named_parameters("")
        .into_iter()
        .zip(["dense.kernel", "dense.bias"])
        .map(|((_, p), hf)| (hf.to_string(), p.shape().to_vec(), p.value()))
        .collect();
    write_f32(&path, &named);

    let ctx2 = Ctx::cpu();
    let dst = Linear::new(&ctx2, 3, 2, 99);
    let map = |s: &str| match s {
        "dense.kernel" => "0".to_string(),
        "dense.bias" => "1".to_string(),
        other => other.to_string(),
    };
    apply_safetensors(&dst, &path, map).unwrap();
    let got: Vec<f32> = dst.parameters().iter().flat_map(|p| p.value()).collect();
    let src_vals: Vec<f32> = src.parameters().iter().flat_map(|p| p.value()).collect();
    assert_eq!(got, src_vals, "name_map must bind HF names to the model FQNs");
    std::fs::remove_file(&path).ok();
}

#[test]
fn apply_is_strict() {
    let path = std::env::temp_dir().join("hodu_st_strict.safetensors");
    let ctx = Ctx::cpu();
    let m = mlp(&ctx, 0);
    let mut rows = model_f32(&m);

    // missing a required model tensor -> Err.
    write_f32(&path, &rows[..rows.len() - 1]);
    assert!(apply_safetensors(&m, &path, |s| s.to_string()).is_err(), "a missing model tensor must Err");

    // an extra file tensor that maps to nothing -> Err.
    rows.push(("junk".to_string(), vec![2], vec![1.0, 2.0]));
    write_f32(&path, &rows);
    assert!(apply_safetensors(&m, &path, |s| s.to_string()).is_err(), "an unmatched file tensor must Err");
    std::fs::remove_file(&path).ok();
}

// Cross-frontend: load the fixture hodu-py writes (known f32 values) and check the bytes agree.
// Skips when the sibling repo is absent, so this crate builds standalone.
#[test]
fn reads_hodu_py_fixture() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/weights.safetensors");
    if !fixture.exists() {
        eprintln!("skip: {} not present", fixture.display());
        return;
    }
    let tensors = load_safetensors(&fixture).unwrap();
    let by_name: std::collections::HashMap<_, _> = tensors.into_iter().collect();
    let a = &by_name["a"];
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.f32(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = &by_name["b"];
    assert_eq!(b.shape, vec![3]);
    assert_eq!(b.f32(), &[0.5, -0.5, 1.25]);
}
