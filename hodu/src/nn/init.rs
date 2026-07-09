//! Deterministic weight initializers (host-side splitmix64; init only, so no need
//! for the engine's counter RNG). Layers pick one at construction.

// splitmix64 uniform draw in [0,1).
fn draw01(s: &mut u64) -> f32 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32 // [0, 1)
}

/// Uniform init in `[-bound, bound]` from a deterministic `seed`.
pub fn uniform(n: usize, bound: f32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| (draw01(&mut s) * 2.0 - 1.0) * bound).collect()
}

/// Xavier/Glorot uniform: bound `sqrt(6/(fan_in+fan_out))`.
pub fn xavier_uniform(n: usize, fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    uniform(n, bound, seed)
}

/// Normal init `N(0, std^2)` (Box-Muller from splitmix64 uniforms).
pub fn normal(n: usize, std: f32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            let u1 = draw01(&mut s).max(1e-7); // avoid ln(0)
            let u2 = draw01(&mut s);
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * std
        })
        .collect()
}

/// Kaiming/He normal: `N(0, 2/fan_in)` (for ReLU nets).
pub fn kaiming_normal(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    normal(n, (2.0 / fan_in as f32).sqrt(), seed)
}
