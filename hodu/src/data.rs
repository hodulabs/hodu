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
mod tests {
    use super::*;

    #[test]
    fn drops_last_and_shapes_batch() {
        // 10 samples of [1,2,2], batch 4 -> 2 full batches (last 2 dropped).
        let x: Vec<f32> = (0..10 * 4).map(|i| i as f32).collect();
        let y: Vec<usize> = (0..10).map(|i| i % 3).collect();
        let ds = Dataset::new(x, vec![1, 2, 2], y).unwrap();
        let mut dl = DataLoader::new(ds, 4, false, 0);
        assert_eq!(dl.len(), 2);
        let bs = dl.batches();
        assert_eq!(bs.len(), 2);
        assert_eq!(bs[0].x_shape, vec![4, 1, 2, 2]);
        assert_eq!(bs[0].x_f32().len(), 4 * 4);
        assert_eq!(bs[0].y_class(), &[0, 1, 2, 0]); // no shuffle -> in order
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let ds = Dataset::new(x, vec![1], (0..8).collect()).unwrap();
        let mut dl = DataLoader::new(ds, 8, true, 7);
        let mut got: Vec<usize> = dl.batches()[0].y_class().to_vec();
        got.sort_unstable();
        assert_eq!(got, (0..8).collect::<Vec<_>>()); // every sample once
    }

    #[test]
    fn tokens_and_regression_and_split() {
        // i64 token inputs batch as I64.
        let ds = Dataset::tokens((0..12).collect(), vec![3], vec![0, 1, 0, 1]).unwrap();
        let mut dl = DataLoader::new(ds, 2, false, 0);
        match &dl.batches()[0].x {
            Data::I64(v) => assert_eq!(v, &[0, 1, 2, 3, 4, 5]),
            _ => panic!("want I64"),
        }
        // regression targets carry their shape; split preserves the sample count.
        let rds = Dataset::regression(
            (0..20).map(|i| i as f32).collect(),
            vec![2],
            (0..10).map(|i| i as f32).collect(),
            vec![1],
        )
        .unwrap();
        let (tr, va) = rds.split(0.8, 1);
        assert_eq!(tr.len() + va.len(), 10);
        assert_eq!(tr.len(), 8);
        let b = &DataLoader::new(tr, 4, false, 0).batches()[0];
        assert!(matches!(b.y, Target::Reg { .. }));
    }

    #[test]
    fn mismatched_data_is_err() {
        // 7 f32 is not a multiple of the 2-wide sample -> Err, not a panic.
        assert!(Dataset::new(vec![0.0; 7], vec![2], vec![0, 1, 2]).is_err());
        // labels != samples (4 samples, 3 labels) -> Err.
        assert!(Dataset::new(vec![0.0; 8], vec![2], vec![0, 1, 0]).is_err());
        // tokens: 5 ids not a multiple of the 2-wide sample -> Err.
        assert!(Dataset::tokens(vec![0i64; 5], vec![2], vec![0, 1]).is_err());
        // tokens: labels (3) != samples (2) -> Err.
        assert!(Dataset::tokens(vec![0i64; 4], vec![2], vec![0, 1, 0]).is_err());
        // regression: targets (3) != samples (2) * per-sample (1) -> Err.
        assert!(Dataset::regression(vec![0.0; 4], vec![2], vec![0.0; 3], vec![1]).is_err());
    }

    #[test]
    fn one_hot_encodes() {
        assert_eq!(one_hot(&[0, 2, 1], 3), vec![1., 0., 0., 0., 0., 1., 0., 1., 0.]);
    }
}
