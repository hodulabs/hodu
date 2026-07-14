//! Linear (32-element affine block) GGML decoders: F32/F16 dense + Q4_0/Q5_0/Q8_0. Each quant
//! block carries one f16 scale; the parent dispatch and shared helpers live in `dequant.rs`.
use super::{f16_to_f32, head, require_blocked};
use std::io;

pub(super) fn f32_raw(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let d = head(data, n * 4, "F32")?;
    Ok(d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub(super) fn f16_raw(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let d = head(data, n * 2, "F16")?;
    Ok(d.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

// Q8_0: 32-element blocks of { d: f16 scale; qs: [i8; 32] } = 34 bytes; x = d * qs[i].
pub(super) fn q8_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
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
pub(super) fn q4_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
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
pub(super) fn q5_0(data: &[u8], n: usize) -> io::Result<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::super::dequant;

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
}
