//! GGML block decoders: a `ggml_type` tag + its raw little-endian bytes -> `n` f32 values.
//! F32/F16 are dense; linear Q4_0/Q5_0/Q8_0 are 32-element affine blocks with an f16 scale;
//! K-quants Q4_K/Q6_K are 256-element super-blocks. Other (super-block) types are not decoded
//! and Err by name so an unsupported-quant file fails loud, not silent.
use crate::serialize::container::inval;
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
fn head<'a>(data: &'a [u8], need: usize, ty: &str) -> io::Result<&'a [u8]> {
    data.get(..need).ok_or_else(|| inval(format!("gguf: {ty} tensor truncated (need {need} bytes)")))
}

fn f32_raw(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let d = head(data, n * 4, "F32")?;
    Ok(d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn f16_raw(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let d = head(data, n * 2, "F16")?;
    Ok(d.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

// Q8_0: 32-element blocks of { d: f16 scale; qs: [i8; 32] } = 34 bytes; x = d * qs[i].
fn q8_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    require_blocked(n, 32, "Q8_0")?;
    let d = head(data, n / 32 * 34, "Q8_0")?;
    let mut out = Vec::with_capacity(n);
    for blk in d.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        out.extend(blk[2..34].iter().map(|&q| scale * q as i8 as f32));
    }
    Ok(out)
}

// Q4_0: 32-element blocks of { d: f16 scale; qs: [u8; 16] } = 18 bytes. Byte j packs two
// quants: low nibble -> x[j], high nibble -> x[j+16]; x = d * (nibble - 8).
fn q4_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    require_blocked(n, 32, "Q4_0")?;
    let d = head(data, n / 32 * 18, "Q4_0")?;
    let mut out = vec![0f32; n];
    for (bi, blk) in d.chunks_exact(18).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let base = bi * 32;
        for (j, &byte) in blk[2..18].iter().enumerate() {
            out[base + j] = scale * ((byte & 0x0f) as i32 - 8) as f32;
            out[base + j + 16] = scale * ((byte >> 4) as i32 - 8) as f32;
        }
    }
    Ok(out)
}

// Q5_0: 32-element blocks of { d: f16 scale; qh: u32 high-bit mask; qs: [u8; 16] } = 22 bytes.
// Byte j packs two 4-bit quants; qh supplies each quant's 5th (high) bit; x = d * (q5 - 16).
fn q5_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    require_blocked(n, 32, "Q5_0")?;
    let d = head(data, n / 32 * 22, "Q5_0")?;
    let mut out = vec![0f32; n];
    for (bi, blk) in d.chunks_exact(22).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let base = bi * 32;
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10; // high bit for elem j    -> bit4
            let xh1 = (qh >> (j + 12)) & 0x10; // high bit for elem j+16 -> bit4
            let x0 = (((qs[j] & 0x0f) as u32 | xh0) as i32) - 16;
            let x1 = (((qs[j] >> 4) as u32 | xh1) as i32) - 16;
            out[base + j] = x0 as f32 * scale;
            out[base + j + 16] = x1 as f32 * scale;
        }
    }
    Ok(out)
}

// ggml get_scale_min_k4: unpack the j-th 6-bit (scale, min) pair from Q4_K's 12 scale bytes.
fn get_scale_min_k4(j: usize, sc: &[u8]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        ((sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))
    }
}

// Q4_K: 256-element super-block { d: f16; dmin: f16; scales: [u8; 12]; qs: [u8; 128] } = 144 bytes.
// Eight 6-bit (scale, min) pairs drive eight 32-quant sub-blocks; x = d*scale*q4 - dmin*min.
fn q4_k(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    require_blocked(n, 256, "Q4_K")?;
    let d_all = head(data, n / 256 * 144, "Q4_K")?;
    let mut out = vec![0f32; n];
    for (bi, blk) in d_all.chunks_exact(144).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let q = &blk[16..144];
        let mut y = bi * 256;
        let mut qoff = 0;
        let mut is = 0;
        for _ in 0..4 {
            let (sc0, mn0) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * sc0 as f32, dmin * mn0 as f32);
            let (sc1, mn1) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * sc1 as f32, dmin * mn1 as f32);
            for l in 0..32 {
                out[y] = d1 * (q[qoff + l] & 0x0f) as f32 - m1;
                y += 1;
            }
            for l in 0..32 {
                out[y] = d2 * (q[qoff + l] >> 4) as f32 - m2;
                y += 1;
            }
            qoff += 32;
            is += 2;
        }
    }
    Ok(out)
}

// Q6_K: 256-element super-block { ql: [u8; 128]; qh: [u8; 64]; scales: [i8; 16]; d: f16 } = 210 bytes.
// Each 6-bit quant = 4 low bits (ql) + 2 high bits (qh); x = d * scale * (q6 - 32).
fn q6_k(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    require_blocked(n, 256, "Q6_K")?;
    let d_all = head(data, n / 256 * 210, "Q6_K")?;
    let mut out = vec![0f32; n];
    for (bi, blk) in d_all.chunks_exact(210).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208]; // signed i8
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let mut y = bi * 256;
        let mut qlo = 0;
        let mut qho = 0;
        let mut sco = 0;
        for _ in 0..2 {
            for l in 0..32 {
                let is = l / 16; // 0 or 1
                let q1 = (((ql[qlo + l] & 0x0f) as i32) | (((qh[qho + l] & 3) as i32) << 4)) - 32;
                let q2 = (((ql[qlo + l + 32] & 0x0f) as i32) | ((((qh[qho + l] >> 2) & 3) as i32) << 4)) - 32;
                let q3 = ((ql[qlo + l] >> 4) as i32 | ((((qh[qho + l] >> 4) & 3) as i32) << 4)) - 32;
                let q4 = ((ql[qlo + l + 32] >> 4) as i32 | ((((qh[qho + l] >> 6) & 3) as i32) << 4)) - 32;
                out[y + l] = d * (scales[sco + is] as i8 as f32) * q1 as f32;
                out[y + l + 32] = d * (scales[sco + is + 2] as i8 as f32) * q2 as f32;
                out[y + l + 64] = d * (scales[sco + is + 4] as i8 as f32) * q3 as f32;
                out[y + l + 96] = d * (scales[sco + is + 6] as i8 as f32) * q4 as f32;
            }
            qlo += 64;
            qho += 32;
            sco += 8;
            y += 128;
        }
    }
    Ok(out)
}

fn require_blocked(n: usize, block: usize, ty: &str) -> io::Result<()> {
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
    fn q8_0_block() {
        // one block: d = 1.0 (f16 bits 0x3C00 -> LE bytes 00 3C), qs = [1, 2, .., 32].
        let mut data = vec![0x00u8, 0x3c];
        data.extend((1..=32i8).map(|q| q as u8));
        let got = dequant(8, &data, 32).unwrap();
        let want: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn q4_0_block() {
        // one block: d = 2.0 (f16 bits 0x4000 -> LE bytes 00 40), qs[j] = j (0..16).
        // low nibble of byte j is j, high nibble is 0. So x[j] = 2*(j-8), x[j+16] = 2*(0-8) = -16.
        let mut data = vec![0x00u8, 0x40];
        data.extend(0u8..16);
        let got = dequant(2, &data, 32).unwrap();
        let mut want = vec![0f32; 32];
        for j in 0..16 {
            want[j] = 2.0 * (j as f32 - 8.0);
            want[j + 16] = -16.0;
        }
        assert_eq!(got, want);
    }

    #[test]
    fn f16_tensor() {
        // two f16 values 1.0, -2.0 -> f32.
        let data = [0x00u8, 0x3c, 0x00, 0xc0];
        assert_eq!(dequant(1, &data, 2).unwrap(), vec![1.0, -2.0]);
    }

    #[test]
    fn k_quant_errs_by_name() {
        // Q5_K (13) is still undecoded; it must fail loud, naming the type.
        let e = dequant(13, &[0u8; 64], 32).unwrap_err();
        assert!(e.to_string().contains("Q5_K"), "unsupported quant error must name the type: {e}");
    }

    // parse an even-length hex string into bytes.
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-5, "idx {i}: got {g}, want {w}");
        }
    }

    // First-block bytes captured from real llama.cpp GGML tensors, cross-checked with gguf-py.
    #[test]
    fn q5_0_llama_bytes() {
        let data = hex("94a6e9b5da145109eb694f4c252360dd38e6d53a1a99");
        let got = dequant(6, &data, 32).unwrap();
        let want = [
            -0.025696, 0.179871, 0.128479, -0.231262, 0.025696, -0.30835, -0.128479, -0.077087, -0.0, 0.077087,
            -0.205566, 0.256958, -0.128479, -0.256958, 0.154175, -0.231262,
        ];
        assert_close(&got[..16], &want);
    }

    #[test]
    fn q4_k_llama_bytes() {
        let data = hex(
            "1814c220ebaef9f3a4aab6f08145edff966bacaea8b99b8c61cc66a28936b6375e87bf6acb2179dc01e9a9829baa50559c6767f7a9088ee6a99fbb66f9c1ca5059adb9766619574caaa35a98f66e39abc547bfe8858403d5ba3d6bc65a86a807028ae5a6908b8bd87ba847c688656e96c7a5a9e585a85486a982f844b3d7c85a48d578c9d408836659b8d08ea897a784",
        );
        let got = dequant(12, &data, 256).unwrap();
        let want = [
            -0.076675, 0.138206, 0.181183, 0.267136, 0.009277, 0.052254, 0.138206, 0.181183, -0.291557, 0.181183,
            -0.076675, -0.248581, 0.052254, -0.076675, -0.076675, -0.033699,
        ];
        assert_close(&got[..16], &want);
    }

    #[test]
    fn q6_k_llama_bytes() {
        let data = hex(
            "196ede1954181be3ec3adcf16cefec907c4d8d045f7b8e12a5f70c2e71d270a2800e5406f9460b966e9021a5b47ecaa020b2abf9a3db41035b5bbde5420eabec94f006d6d064b4e04de1ddf765c50301c3e143169f86ffb10649b65f76b8064256587004c7be82b43116e5f3d1e444b93402fb8978cdd2b400cfc1f0a271768e8ba8c5a72a6b19b5643596a79640c640a2ea990aa5759df95a86e9a9641d9a123898aa1698e5e205aa69545a1aab8eb092296b66505a98448cb56711675e9563d48045cbd455cec929c049e2c5412855e80d",
        );
        let got = dequant(14, &data, 256).unwrap();
        let want = [
            -0.396538, 0.285507, 0.031723, -0.396538, -0.063446, -0.380676, 0.079308, 0.2062, 0.31723, 0.095169,
            -0.190338, -0.269646, -0.190338, 0.269646, -0.190338, 0.507568,
        ];
        assert_close(&got[..16], &want);
    }

    #[test]
    fn truncated_and_unaligned_err() {
        assert!(dequant(8, &[0u8; 10], 32).is_err(), "short Q8_0 buffer must Err");
        assert!(dequant(8, &[0u8; 34], 16).is_err(), "non-multiple-of-32 Q8_0 must Err");
    }
}
