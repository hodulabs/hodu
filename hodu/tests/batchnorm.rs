//! BatchNorm on the static graph: train normalizes by batch stats, eval by the
//! host-EMA running stats (one graph, blended by the fed train flag).
use hodu::prelude::*;

// (a) train mode: each channel is normalized to ~0 mean / ~1 var (biased).
#[test]
fn train_normalizes_per_channel() {
    let ctx = Ctx::cpu();
    let bn = BatchNorm1d::new(&ctx, 2, 1e-5, 0.1);
    let x = ctx.input(vec![4, 2]);
    // rows (c0,c1): c0=[1,2,3,4], c1=[10,20,30,40]
    ctx.feed(x.node(), vec![1., 10., 2., 20., 3., 30., 4., 40.], vec![4, 2]);
    let y = bn.forward(&x).unwrap().realize();
    for c in 0..2 {
        let col: Vec<f32> = (0..4).map(|r| y[r * 2 + c]).collect();
        let mean = col.iter().sum::<f32>() / 4.0;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3, "channel {c} mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "channel {c} var {var}");
    }
}

// (b) eval uses the running stats, not the current batch. Running is set to batch A
// (momentum 1.0); eval on a constant batch B must normalize by A -> nonzero output
// (batch-stat normalization of a constant B would give ~0).
#[test]
fn eval_uses_running_stats() {
    let ctx = Ctx::cpu();
    let bn = BatchNorm1d::new(&ctx, 2, 1e-5, 1.0);
    let x = ctx.input(vec![4, 2]);
    let y = bn.forward(&x).unwrap();
    // A: c0=[-1,1,-1,1] mean 0 var 1 ; c1=[-2,2,-2,2] mean 0 var 4
    ctx.feed(x.node(), vec![-1., -2., 1., 2., -1., -2., 1., 2.], vec![4, 2]);
    let _ = y.realize(); // train forward
    bn.update_running(); // running = A stats exactly (momentum 1.0)

    // B: constant 10 / 20
    ctx.feed(x.node(), vec![10., 20., 10., 20., 10., 20., 10., 20.], vec![4, 2]);
    ctx.set_training(false);
    let out = y.realize(); // eval: (B - runmean)/sqrt(runvar+eps)
    for r in 0..4 {
        // c0 = (10-0)/sqrt(1) = 10 ; c1 = (20-0)/sqrt(4) = 10
        assert!((out[r * 2] - 10.0).abs() < 0.1, "c0 {}", out[r * 2]);
        assert!((out[r * 2 + 1] - 10.0).abs() < 0.1, "c1 {}", out[r * 2 + 1]);
    }
}

// (c) an MLP with BatchNorm trains (loss drops) and eval-mode forward runs.
#[test]
fn mlp_with_batchnorm_trains() {
    let ctx = Ctx::cpu();
    let (xs, labels) = blobs();
    let n = labels.len();
    let x = ctx.input(vec![n, 4]);
    ctx.feed(x.node(), xs, vec![n, 4]);
    let targets = ctx.input(vec![n, 3]);
    ctx.feed(targets.node(), one_hot(&labels, 3), vec![n, 3]);

    let l1 = Linear::new(&ctx, 4, 16, 1);
    let bn = BatchNorm1d::new(&ctx, 16, 1e-5, 0.1);
    let l2 = Linear::new(&ctx, 16, 3, 2);
    let h = bn.forward(&l1.forward(&x).unwrap()).unwrap().relu();
    let logits = l2.forward(&h).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    let l0 = loss.item();

    let mut params = l1.parameters();
    params.extend(bn.parameters());
    params.extend(l2.parameters());
    let pts: Vec<&Tensor> = params.iter().map(Param::tensor).collect();
    let grads = loss.grad(&pts).unwrap();
    let mut opt = Adam::new(params.clone(), 0.05);
    for _ in 0..200 {
        opt.step(&grad_values(&grads));
        bn.update_running();
    }
    assert!(loss.item() < l0 * 0.5, "loss did not drop: {l0} -> {}", loss.item());

    ctx.set_training(false);
    let _ = logits.realize(); // eval-mode forward runs without panic
}

// The base BatchNorm (prelude-exported, no rank wrapper) Errs on a rank-1 input rather
// than panicking on the channel-axis index its reductions assume.
#[test]
fn base_batchnorm_rank1_errors() {
    let ctx = Ctx::cpu();
    let bn = BatchNorm::new(&ctx, 4, 1e-5, 0.1);
    assert!(bn.forward(&ctx.zeros(vec![4])).is_err());
}

// 3 gaussian blobs in 4-D (channels 0,1 informative, 2,3 noise), class = which blob.
fn blobs() -> (Vec<f32>, Vec<usize>) {
    let centers = [(2.5f32, 2.5f32), (-2.5, 2.5), (0.0, -2.5)];
    let mut s = 999u64;
    let mut jit = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 1.6 - 0.8
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (c, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..60 {
            xs.push(cx + jit());
            xs.push(cy + jit());
            xs.push(jit()); // noise
            xs.push(jit());
            ys.push(c);
        }
    }
    (xs, ys)
}
