//! Weight-only `QuantLinear` (the deploy path): int8/int4 group-wise quant matches the
//! f32 `Linear` within quant error and the `dequant_matmul` CPU reference exactly, and
//! its packed U8 weight + f16 scales survive a `.hodu` save/load round-trip.
use hodu::kurumi::{dequant_matmul, quantize};
use hodu::prelude::*;

// deterministic xorshift pseudo-random in [-1, 1).
fn prng(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

// relative L1 error, guarded against a zero reference.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    let den: f32 = b.iter().map(|x| x.abs()).sum::<f32>().max(1e-6);
    num / den
}

const IN: usize = 64;
const OUT: usize = 16;
const M: usize = 4;
const GS: usize = 32; // IN is a multiple of GS

// f32 Linear with a nonzero bias (so the broadcast add is exercised).
fn linear_with_bias(ctx: &Ctx, seed: u64) -> Linear {
    let lin = Linear::new(ctx, IN, OUT, seed);
    lin.bias().set(prng(seed ^ 0xB1A5, OUT));
    lin
}

// dequant_matmul(act, transpose-and-quantize(lin.weight)) + bias -- the CPU reference
// QuantLinear should reproduce.
fn reference(lin: &Linear, act: &[f32], bits: u8, symmetric: bool) -> Vec<f32> {
    let wv = lin.weight().value(); // [IN, OUT] row-major
    let mut wt = vec![0f32; OUT * IN];
    for i in 0..IN {
        for j in 0..OUT {
            wt[j * IN + i] = wv[i * OUT + j];
        }
    }
    let q = quantize(&wt, OUT, IN, bits, GS, symmetric);
    let mut out = dequant_matmul(act, M, &q); // [M, OUT]
    let bias = lin.bias().value();
    for mi in 0..M {
        for j in 0..OUT {
            out[mi * OUT + j] += bias[j];
        }
    }
    out
}

fn accuracy(bits: u8, symmetric: bool, tol_vs_f32: f32) {
    let ctx = Ctx::cpu();
    let lin = linear_with_bias(&ctx, 7);
    let act_v = prng(3, M * IN);
    let act = ctx.constant(act_v.clone(), vec![M, IN]);

    let f32_out = lin.forward(&act).unwrap().realize();
    let ql = QuantLinear::from_linear(&lin, bits, GS, symmetric).unwrap();
    let q_out = ql.forward(&act).unwrap().realize();
    let ref_dq = reference(&lin, &act_v, bits, symmetric);

    let e_ref = rel_err(&q_out, &ref_dq);
    let e_f32 = rel_err(&q_out, &f32_out);
    assert!(e_ref < 1e-3, "bits={bits} sym={symmetric}: vs dequant_matmul rel err {e_ref}");
    assert!(e_f32 < tol_vs_f32, "bits={bits} sym={symmetric}: vs f32 Linear rel err {e_f32} (tol {tol_vs_f32})");
}

#[test]
fn int8_symmetric_matches() {
    accuracy(8, true, 0.05);
}

#[test]
fn int8_asymmetric_matches() {
    accuracy(8, false, 0.05);
}

#[test]
fn int4_symmetric_matches() {
    accuracy(4, true, 0.4);
}

#[test]
fn int4_asymmetric_matches() {
    accuracy(4, false, 0.4);
}

#[test]
fn group_size_must_divide_in() {
    let ctx = Ctx::cpu();
    let lin = Linear::new(&ctx, IN, OUT, 1);
    assert!(QuantLinear::from_linear(&lin, 8, 48, true).is_err(), "in=64 not a multiple of group_size=48 must error");
}

// save a Sequential holding a QuantLinear, load into one built from DIFFERENT weights,
// and assert the forward output is reproduced bit-exactly -- the packed/scales/mins
// buffers survived the container (nested byte-buffer aggregation included).
#[test]
fn round_trip_preserves_forward() {
    let path = std::env::temp_dir().join("hodu_quant_roundtrip.hodu");
    let act_v = prng(3, M * IN);

    let ctx = Ctx::cpu();
    let ql = QuantLinear::from_linear(&linear_with_bias(&ctx, 7), 4, GS, false).unwrap();
    let model = Sequential::new(vec![Box::new(ql)]);
    let act = ctx.constant(act_v.clone(), vec![M, IN]);
    let want = model.forward(&act).unwrap().realize();
    save(&path, &model).unwrap();

    let ctx2 = Ctx::cpu();
    let ql2 = QuantLinear::from_linear(&linear_with_bias(&ctx2, 123), 4, GS, false).unwrap();
    let fresh = Sequential::new(vec![Box::new(ql2)]);
    let act2 = ctx2.constant(act_v, vec![M, IN]);
    let before = fresh.forward(&act2).unwrap().realize();
    assert_ne!(before, want, "different weights should differ before load");

    load(&path, &fresh).unwrap();
    let after = fresh.forward(&act2).unwrap().realize();
    assert_eq!(after, want, "loaded quant weights must reproduce the saved forward exactly");
    std::fs::remove_file(&path).ok();
}

// Same round-trip but SYMMETRIC (mins=None): the 1-buffer path (scales only, no mins).
// The asymmetric case above has 2 buffers; this pins that the single-buffer layout also
// survives save -> fresh -> load and reproduces the forward exactly.
#[test]
fn symmetric_round_trip_preserves_forward() {
    let path = std::env::temp_dir().join("hodu_quant_sym_roundtrip.hodu");
    let act_v = prng(3, M * IN);

    let ctx = Ctx::cpu();
    let ql = QuantLinear::from_linear(&linear_with_bias(&ctx, 7), 4, GS, true).unwrap();
    let model = Sequential::new(vec![Box::new(ql)]);
    let act = ctx.constant(act_v.clone(), vec![M, IN]);
    let want = model.forward(&act).unwrap().realize();
    save(&path, &model).unwrap();

    let ctx2 = Ctx::cpu();
    let ql2 = QuantLinear::from_linear(&linear_with_bias(&ctx2, 123), 4, GS, true).unwrap();
    let fresh = Sequential::new(vec![Box::new(ql2)]);
    let act2 = ctx2.constant(act_v, vec![M, IN]);
    let before = fresh.forward(&act2).unwrap().realize();
    assert_ne!(before, want, "different weights should differ before load");

    load(&path, &fresh).unwrap();
    let after = fresh.forward(&act2).unwrap().realize();
    assert_eq!(after, want, "symmetric (mins=None) quant weights must round-trip through save/load");
    std::fs::remove_file(&path).ok();
}
