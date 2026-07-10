//! In-memory dataset + mini-batch loader for the static / feed training model.
//! The loss graph is built once for a fixed batch shape, so the loader drops the
//! tail partial batch -- every fed batch is exactly `batch` wide.
//!
//! Inputs are f32 features OR i64 token ids (for `Embedding`); targets are class
//! labels OR f32 regression values. A `Batch` feeds itself into an Input via
//! [`Batch::feed_x`] / [`Batch::feed_y`], picking `feed` vs `feed_i64` by kind.
//!
//! Static graph => fixed batch shape => drop_last. A ragged final batch
//! would need a second graph; not worth it for training.
mod dataset;
mod loader;
mod types;

pub use dataset::Dataset;
pub use loader::{Batch, DataLoader};
pub use types::{Data, Target};

// Copy rows `idx` (each `per` wide) out of a flat row-major buffer.
fn gather_flat<T: Copy>(src: &[T], per: usize, idx: &[usize]) -> Vec<T> {
    let mut out = Vec::with_capacity(idx.len() * per);
    for &s in idx {
        out.extend_from_slice(&src[s * per..(s + 1) * per]);
    }
    out
}

/// One-hot encode `labels` into a flat `[labels.len(), classes]` f32 tensor, the
/// target format `hodu::loss::cross_entropy` expects.
pub fn one_hot(labels: &[usize], classes: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; labels.len() * classes];
    for (i, &c) in labels.iter().enumerate() {
        o[i * classes + c] = 1.0;
    }
    o
}

// splitmix64: advance the counter by the golden gamma, return the mixed draw.
fn draw(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests;
