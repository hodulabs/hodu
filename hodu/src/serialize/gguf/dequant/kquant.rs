//! K-quant (256-element super-block) GGML decoders: Q4_K/Q6_K. Each super-block packs several
//! 6-bit (scale, min) pairs over its sub-blocks; parent dispatch + shared helpers in `dequant.rs`.
use super::{f16_to_f32, head, require_blocked};
use std::io;

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
pub(super) fn q4_k(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
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
pub(super) fn q6_k(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::super::dequant;

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
}
