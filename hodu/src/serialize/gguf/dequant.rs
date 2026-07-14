//! GGML block decoders: a `ggml_type` tag + its raw little-endian bytes -> `n` f32 values.
//! F32/F16 are dense; linear Q4_0/Q5_0/Q8_0 are 32-element affine blocks with an f16 scale;
//! K-quants Q4_K/Q6_K are 256-element super-blocks. Other (super-block) types are not decoded
//! and Err by name so an unsupported-quant file fails loud, not silent. This parent holds the
//! dispatch + shared helpers; the decoders live in `linear` and `kquant`.
mod kquant;
mod linear;

use crate::serialize::container::inval;
use kquant::{q4_k, q6_k};
use linear::{f16_raw, f32_raw, q4_0, q5_0, q8_0};
use std::io;

// IEEE-754 half -> f32. No `half` dep: block scales and F16 tensors decode here by hand.
pub(super) fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as f32;
    let val = match exp {
        0 => mant * 2f32.powi(-24),                              // subnormal / zero
        0x1f if mant == 0.0 => f32::INFINITY,                    // inf
        0x1f => f32::NAN,                                        // nan
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15), // normal
    };
    sign * val
}

/// Decode a tensor of `n` elements stored as `ggml_type` from the start of `data` into f32.
/// Errors (naming the type) on any type this doesn't decode, so unsupported quants fail loud.
pub(super) fn dequant(ggml_type: u32, data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    match ggml_type {
        0 => f32_raw(data, n),
        1 => f16_raw(data, n),
        8 => q8_0(data, n),
        2 => q4_0(data, n),
        6 => q5_0(data, n),
        12 => q4_k(data, n),
        14 => q6_k(data, n),
        t => Err(inval(format!("gguf: ggml type {t} ({}) not supported", type_name(t)))),
    }
}

// Slice exactly `need` bytes off the front of a tensor's data, or Err naming the type.
pub(super) fn head<'a>(data: &'a [u8], need: usize, ty: &str) -> io::Result<&'a [u8]> {
    data.get(..need).ok_or_else(|| inval(format!("gguf: {ty} tensor truncated (need {need} bytes)")))
}

pub(super) fn require_blocked(n: usize, block: usize, ty: &str) -> io::Result<()> {
    if !n.is_multiple_of(block) {
        return Err(inval(format!("gguf: {ty} tensor len {n} is not a multiple of the {block}-element block")));
    }
    Ok(())
}

// ggml_type -> name, so an undecoded (K-quant) type Errs by name, not just a number.
fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_one() {
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 0x3C00 is the half bit-pattern for 1.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xc000), -2.0); // sign + exp for -2.0
    }

    #[test]
    fn k_quant_errs_by_name() {
        // Q5_K (13) is still undecoded; it must fail loud, naming the type.
        let e = dequant(13, &[0u8; 64], 32).unwrap_err();
        assert!(e.to_string().contains("Q5_K"), "unsupported quant error must name the type: {e}");
    }

    #[test]
    fn truncated_and_unaligned_err() {
        assert!(dequant(8, &[0u8; 10], 32).is_err(), "short Q8_0 buffer must Err");
        assert!(dequant(8, &[0u8; 34], 16).is_err(), "non-multiple-of-32 Q8_0 must Err");
    }
}
