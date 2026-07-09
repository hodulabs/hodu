//! `.hodu` v1 container: named params + buffers round-trip, self-describing load
//! (mismatch errors clearly), BatchNorm running stats survive save/load (the eval
//! correctness bug), and optimizer state resumes a training run.
use hodu::prelude::*;

fn mlp(ctx: &Ctx, seed: u64) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(ctx, 3, 4, seed)),
        Box::new(Relu),
        Box::new(Linear::new(ctx, 4, 2, seed + 1)),
    ])
}

#[test]
fn hodu_save_load_round_trip() {
    let path = std::env::temp_dir().join("hodu_roundtrip_test.hodu");

    // "trained" model (one init) and its output on a fixed input.
    let ctx = Ctx::cpu();
    let trained = mlp(&ctx, 0);
    let x = ctx.constant(vec![0.5, -1.0, 2.0], vec![1, 3]);
    let want = trained.forward(&x).unwrap().realize();
    save(&path, &trained).unwrap();

    // a fresh model with a DIFFERENT init -> different output.
    let ctx2 = Ctx::cpu();
    let fresh = mlp(&ctx2, 100);
    let x2 = ctx2.constant(vec![0.5, -1.0, 2.0], vec![1, 3]);
    let before = fresh.forward(&x2).unwrap().realize();
    assert_ne!(before, want, "different init should differ before load");

    // load restores the weights (by name) -> bit-exact same output.
    load(&path, &fresh).unwrap();
    let after = fresh.forward(&x2).unwrap().realize();
    assert_eq!(after, want, "loaded weights must reproduce the saved model exactly");

    std::fs::remove_file(&path).ok();
}

#[test]
fn load_rejects_mismatched_architecture() {
    let path = std::env::temp_dir().join("hodu_mismatch_test.hodu");
    let ctx = Ctx::cpu();
    save(&path, &mlp(&ctx, 0)).unwrap();

    let ctx2 = Ctx::cpu();
    let wrong = Sequential::new(vec![Box::new(Linear::new(&ctx2, 3, 8, 0))]); // wrong shapes/count
    assert!(load(&path, &wrong).is_err(), "name/shape mismatch must error, not silently corrupt");
    std::fs::remove_file(&path).ok();
}

// A small Linear -> BatchNorm -> Relu -> Linear net that owns a BatchNorm handle (so
// the test can drive `update_running`). Uses the default `named_*` (flat numbering).
struct BnNet {
    l1: Linear,
    bn: BatchNorm1d,
    l2: Linear,
}

impl BnNet {
    fn new(ctx: &Ctx, seed: u64) -> BnNet {
        BnNet {
            l1: Linear::new(ctx, 4, 8, seed),
            bn: BatchNorm1d::new(ctx, 8, 1e-5, 0.5),
            l2: Linear::new(ctx, 8, 3, seed + 1),
        }
    }
}

impl Module for BnNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let h = self.bn.forward(&self.l1.forward(x)?)?.relu();
        self.l2.forward(&h)
    }
    fn parameters(&self) -> Vec<Param> {
        let mut p = self.l1.parameters();
        p.extend(self.bn.parameters());
        p.extend(self.l2.parameters());
        p
    }
    fn buffers(&self) -> Vec<Buffer> {
        self.bn.buffers()
    }
}

// THE bug fix: BatchNorm running stats are buffers, not params. Before the fix,
// save/load dropped them -> a reloaded model ran eval with running stats at init
// (0/1), silently wrong. This pins that they persist: a fresh model that loads the
// checkpoint must reproduce the trained model's eval-mode output bit-for-bit.
#[test]
fn batchnorm_running_stats_survive_round_trip() {
    let path = std::env::temp_dir().join("hodu_bn_roundtrip.hodu");
    let n = 16usize;
    let train_batch: Vec<f32> = (0..n * 4).map(|i| ((i * 7 % 13) as f32) - 3.0).collect();
    let eval_batch: Vec<f32> = (0..n * 4).map(|i| ((i % 5) as f32) * 0.5).collect();

    // train the running stats off their (0,1) init, then record the eval-mode output.
    let ctx = Ctx::cpu();
    let net = BnNet::new(&ctx, 1);
    let x = ctx.input(vec![n, 4]);
    ctx.feed(x.node(), train_batch, vec![n, 4]);
    let y = net.forward(&x).unwrap();
    for _ in 0..10 {
        let _ = y.realize(); // train-mode forward on the fed batch
        net.bn.update_running(); // EMA the running stats from that batch
    }
    ctx.set_training(false);
    ctx.feed(x.node(), eval_batch.clone(), vec![n, 4]);
    let want = y.realize(); // eval: normalized by running stats
    save(&path, &net).unwrap();

    // a FRESH net: different init, running stats at (0,1). load restores BOTH weights
    // and running stats -> identical eval output.
    let ctx2 = Ctx::cpu();
    let fresh = BnNet::new(&ctx2, 99);
    let x2 = ctx2.input(vec![n, 4]);
    ctx2.feed(x2.node(), eval_batch, vec![n, 4]);
    ctx2.set_training(false);
    let y2 = fresh.forward(&x2).unwrap();
    let before = y2.realize();
    assert_ne!(before, want, "different init/running stats should differ before load");

    load(&path, &fresh).unwrap();
    let after = y2.realize();
    assert_eq!(after, want, "BatchNorm running stats must persist across save/load (eval-mode correctness)");
    std::fs::remove_file(&path).ok();
}

// A fixed-data Adam training harness so a run is fully deterministic given the
// (params + optimizer state).
struct Trainer {
    model: Sequential,
    opt: Adam,
    grads: Vec<Tensor>,
}

impl Trainer {
    fn new(seed: u64) -> Trainer {
        let ctx = Ctx::cpu();
        let model = mlp(&ctx, seed);
        let x = ctx.input(vec![4, 3]);
        ctx.feed(x.node(), vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 1.0, -1.1, 1.2], vec![4, 3]);
        let target = ctx.input(vec![4, 2]);
        ctx.feed(target.node(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![4, 2]);
        let loss = mse_loss(&model.forward(&x).unwrap(), &target).unwrap();
        let params = model.parameters();
        let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
        let grads = loss.grad(&pts).unwrap();
        let opt = Adam::new(params, 0.05);
        Trainer { model, opt, grads }
    }
    fn step_n(&mut self, k: usize) {
        for _ in 0..k {
            self.opt.step(&grad_values(&self.grads));
        }
    }
    fn params(&self) -> Vec<f32> {
        self.model.parameters().iter().flat_map(|p| p.value()).collect()
    }
}

// Optimizer resume: N steps + checkpoint + load into fresh model+opt + M more steps
// must equal an uninterrupted N+M run (Adam moments and step count restored).
#[test]
fn optimizer_state_resumes_training() {
    let path = std::env::temp_dir().join("hodu_ckpt.hodu");
    let (n_first, n_more) = (5usize, 7usize);

    // reference: one uninterrupted run of N+M steps.
    let mut reference = Trainer::new(0);
    reference.step_n(n_first + n_more);
    let want = reference.params();

    // split run: N steps, then checkpoint model + optimizer.
    let mut a = Trainer::new(0);
    a.step_n(n_first);
    save_checkpoint(&path, &a.model, &a.opt).unwrap();

    // fresh model+opt (different init, zero moments): load restores params + Adam
    // state, then M more steps should catch up to the uninterrupted run.
    let mut b = Trainer::new(123);
    load_checkpoint(&path, &b.model, &mut b.opt).unwrap();
    b.step_n(n_more);
    let got = b.params();

    assert_eq!(got.len(), want.len(), "param vector size");
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-6, "resumed param {g} != uninterrupted {w}");
    }

    // a checkpoint still loads as a plain model (optimizer rows ignored).
    let ctx = Ctx::cpu();
    let plain = mlp(&ctx, 7);
    load(&path, &plain).unwrap();

    std::fs::remove_file(&path).ok();
}

// SGD-with-momentum resume: the Adam test above covers moments+step; this pins that
// SGD's velocity (`vel.*`) round-trips. Build a fixed harness (model + grads + Sgd),
// run N+M uninterrupted for the reference, then N -> checkpoint -> fresh model+Sgd load
// -> M more. Without restored velocity the momentum term would reset to 0 at the resume
// and the split run would diverge from the reference.
fn sgd_harness(seed: u64) -> (Sequential, Vec<Tensor>, Sgd) {
    let ctx = Ctx::cpu();
    let model = mlp(&ctx, seed);
    let x = ctx.input(vec![4, 3]);
    ctx.feed(x.node(), vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 1.0, -1.1, 1.2], vec![4, 3]);
    let target = ctx.input(vec![4, 2]);
    ctx.feed(target.node(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![4, 2]);
    let loss = mse_loss(&model.forward(&x).unwrap(), &target).unwrap();
    let params = model.parameters();
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let opt = Sgd::with_momentum(params, 0.05, 0.9, 0.0);
    (model, grads, opt)
}

fn param_snapshot(model: &Sequential) -> Vec<f32> {
    model.parameters().iter().flat_map(|p| p.value()).collect()
}

#[test]
fn sgd_momentum_state_resumes_training() {
    let path = std::env::temp_dir().join("hodu_sgd_ckpt.hodu");
    let (n_first, n_more) = (5usize, 7usize);
    let run = |opt: &Sgd, grads: &[Tensor], k: usize| {
        for _ in 0..k {
            opt.step(&grad_values(grads));
        }
    };

    // reference: one uninterrupted N+M run.
    let (ref_model, ref_grads, ref_opt) = sgd_harness(0);
    run(&ref_opt, &ref_grads, n_first + n_more);
    let want = param_snapshot(&ref_model);

    // split run: N steps, then checkpoint model + Sgd velocity.
    let (a_model, a_grads, a_opt) = sgd_harness(0);
    run(&a_opt, &a_grads, n_first);
    save_checkpoint(&path, &a_model, &a_opt).unwrap();

    // fresh model+Sgd (different init, zero velocity): load restores params + vel, then
    // M more steps must catch up to the uninterrupted run.
    let (b_model, b_grads, mut b_opt) = sgd_harness(123);
    load_checkpoint(&path, &b_model, &mut b_opt).unwrap();
    run(&b_opt, &b_grads, n_more);
    let got = param_snapshot(&b_model);

    assert_eq!(got.len(), want.len(), "param vector size");
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-6, "resumed param {g} != uninterrupted {w}");
    }
    std::fs::remove_file(&path).ok();
}

// load_checkpoint must Err (not silently skip) when the optimizer state is absent or
// doesn't fit the optimizer -- the `take_slot` error arms.
#[test]
fn checkpoint_missing_or_mismatched_optim_errors() {
    let path = std::env::temp_dir().join("hodu_ckpt_bad.hodu");

    // a PLAIN model file (no optim rows). The model loads, but Adam finds no "step" ->
    // load_checkpoint Errs.
    let a = Trainer::new(0);
    save(&path, &a.model).unwrap();
    let mut b = Trainer::new(1);
    assert!(
        load_checkpoint(&path, &b.model, &mut b.opt).is_err(),
        "a plain file has no optimizer state -> load_checkpoint must Err"
    );

    // a real Adam checkpoint, loaded into an Sgd over the SAME model: the model matches
    // but Sgd expects "vel.*" rows an Adam checkpoint never wrote -> Err.
    save_checkpoint(&path, &a.model, &a.opt).unwrap();
    let c = Trainer::new(2);
    let mut sgd = Sgd::new(c.model.parameters(), 0.1);
    assert!(
        load_checkpoint(&path, &c.model, &mut sgd).is_err(),
        "Adam checkpoint into Sgd (different optim slots) must Err"
    );

    std::fs::remove_file(&path).ok();
}

// A custom container that implements ONLY `children()` (the blessed API) and nests a
// QuantLinear. This pins the recursion trap: the packed U8 byte-buffer is reachable
// only through the container's children walk, so if the derived named_byte_buffers ever
// stopped recursing children, the quant weight would silently drop and forward would
// diverge after a round-trip.
struct QNet {
    l1: Linear,
    q: QuantLinear,
}

impl Module for QNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        self.q.forward(&self.l1.forward(x)?)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("l1".to_string(), &self.l1 as &dyn Module), ("q".to_string(), &self.q)]
    }
}

fn qnet(ctx: &Ctx, seed: u64) -> QNet {
    let l1 = Linear::new(ctx, 4, 32, seed);
    let src = Linear::new(ctx, 32, 8, seed ^ 0x55);
    let q = QuantLinear::from_linear(&src, 4, 16, false).unwrap();
    QNet { l1, q }
}

#[test]
fn quant_byte_buffer_survives_custom_container() {
    let path = std::env::temp_dir().join("hodu_qnet_roundtrip.hodu");
    let ctx = Ctx::cpu();
    let net = qnet(&ctx, 1);
    let x = ctx.constant((0..4 * 4).map(|i| (i as f32) * 0.1 - 0.7).collect(), vec![4, 4]);
    let want = net.forward(&x).unwrap().realize();
    save(&path, &net).unwrap();

    let ctx2 = Ctx::cpu();
    let fresh = qnet(&ctx2, 999);
    let x2 = ctx2.constant((0..4 * 4).map(|i| (i as f32) * 0.1 - 0.7).collect(), vec![4, 4]);
    let before = fresh.forward(&x2).unwrap().realize();
    assert_ne!(before, want, "different init should differ before load");

    load(&path, &fresh).unwrap();
    let after = fresh.forward(&x2).unwrap().realize();
    assert_eq!(after, want, "quant byte-buffer must persist through a custom children() container");
    std::fs::remove_file(&path).ok();
}

// A TransformerEncoder-based model (custom children() container) round-trips: the
// per-block-index + ln1./attn.q./ff1. FQNs must be stable, or load-by-name fails.
struct TinyXf {
    emb: Embedding,
    enc: TransformerEncoder,
    head: Linear,
}

impl TinyXf {
    fn new(ctx: &Ctx, seed: u64) -> TinyXf {
        TinyXf {
            emb: Embedding::new(ctx, 6, 8, seed),
            enc: TransformerEncoder::new(ctx, 8, 2, 2, false, true, seed ^ 0x9).unwrap(),
            head: Linear::new(ctx, 8, 3, seed ^ 0x7),
        }
    }
    // token idx [B,S] -> [B, CLASSES] via emb -> encoder -> mean-pool -> head.
    fn logits(&self, idx: &Tensor) -> Tensor {
        let h = self.emb.forward(idx).unwrap();
        let h = self.enc.forward(&h).unwrap();
        self.head.forward(&h.mean_axis(1).unwrap()).unwrap()
    }
}

impl Module for TinyXf {
    fn forward(&self, idx: &Tensor) -> Result<Tensor, Error> {
        Ok(self.logits(idx))
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("emb".to_string(), &self.emb as &dyn Module),
            ("enc".to_string(), &self.enc),
            ("head".to_string(), &self.head),
        ]
    }
}

#[test]
fn transformer_fqn_round_trip() {
    let path = std::env::temp_dir().join("hodu_xf_roundtrip.hodu");
    let ids: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 1, 0]; // [2, 4]

    let ctx = Ctx::cpu();
    let trained = TinyXf::new(&ctx, 3);
    let idx = ctx.input_i64(vec![2, 4]);
    ctx.feed_i64(idx.node(), ids.clone(), vec![2, 4]);
    let want = trained.forward(&idx).unwrap().realize();
    save(&path, &trained).unwrap();

    let ctx2 = Ctx::cpu();
    let fresh = TinyXf::new(&ctx2, 100);
    let idx2 = ctx2.input_i64(vec![2, 4]);
    ctx2.feed_i64(idx2.node(), ids, vec![2, 4]);
    let before = fresh.forward(&idx2).unwrap().realize();
    assert_ne!(before, want, "different init should differ before load");

    load(&path, &fresh).unwrap();
    let after = fresh.forward(&idx2).unwrap().realize();
    assert_eq!(after, want, "TransformerEncoder FQNs must round-trip (q./k./ln1./per-block naming stable)");
    std::fs::remove_file(&path).ok();
}
