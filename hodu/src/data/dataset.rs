//! The in-memory `Dataset`: a flat row-major sample buffer + labels, with the
//! shape-invariant checks (`new`/`tokens`/`regression`/`build`) and a deterministic
//! `split`. The `DataLoader` (see loader.rs) batches over it.
use crate::data::{Data, Target, draw, gather_flat};
use hodu_core::Error;

fn data_err(msg: String) -> Error {
    Error::Shape { op: "Dataset", msg }
}

/// A flat sample/label dataset. Input is row-major over `[len, sample_shape...]`.
/// Fields are `pub(super)` so the sibling `DataLoader` can batch over them.
pub struct Dataset {
    pub(super) x: Data,
    pub(super) sample_shape: Vec<usize>,
    pub(super) y: Target,
    pub(super) len: usize,
}

impl Dataset {
    /// f32 features + integer class labels.
    pub fn new(x: Vec<f32>, sample_shape: Vec<usize>, y: Vec<usize>) -> Result<Dataset, Error> {
        Dataset::build(Data::F32(x), sample_shape, Target::Class(y))
    }

    /// i64 token ids + integer class labels (for `Embedding`-fronted models).
    pub fn tokens(x: Vec<i64>, sample_shape: Vec<usize>, y: Vec<usize>) -> Result<Dataset, Error> {
        Dataset::build(Data::I64(x), sample_shape, Target::Class(y))
    }

    /// f32 features + f32 regression targets (`target_shape` is per-sample).
    pub fn regression(
        x: Vec<f32>,
        sample_shape: Vec<usize>,
        y: Vec<f32>,
        target_shape: Vec<usize>,
    ) -> Result<Dataset, Error> {
        Dataset::build(Data::F32(x), sample_shape, Target::Reg { data: y, shape: target_shape })
    }

    fn build(x: Data, sample_shape: Vec<usize>, y: Target) -> Result<Dataset, Error> {
        let per: usize = sample_shape.iter().product();
        let xlen = match &x {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
        };
        let len = xlen / per;
        if xlen != len * per {
            return Err(data_err(format!("x len {xlen} not a multiple of {per}")));
        }
        match &y {
            Target::Class(labels) => {
                if labels.len() != len {
                    return Err(data_err(format!("labels {} != samples {len}", labels.len())));
                }
            }
            Target::Reg { data, shape } => {
                let tp: usize = shape.iter().product();
                if data.len() != len * tp {
                    return Err(data_err(format!("reg targets {} != samples {len} * {tp}", data.len())));
                }
            }
        }
        Ok(Dataset { x, sample_shape, y, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn sample_size(&self) -> usize {
        self.sample_shape.iter().product()
    }

    /// Deterministically split into `(train, val)` by `train_frac` of the samples
    /// (shuffled by `seed`). Used to avoid hand-building two loaders.
    pub fn split(self, train_frac: f32, seed: u64) -> (Dataset, Dataset) {
        let mut idx: Vec<usize> = (0..self.len).collect();
        let mut rng = seed ^ 0x2545_F491_4F6C_DD1D;
        for i in (1..idx.len()).rev() {
            idx.swap(i, (draw(&mut rng) >> 33) as usize % (i + 1));
        }
        let n_train = (self.len as f32 * train_frac).round() as usize;
        let (a, b) = idx.split_at(n_train);
        (self.gather(a), self.gather(b))
    }

    // Build a sub-dataset from sample indices (copies the selected rows).
    fn gather(&self, idx: &[usize]) -> Dataset {
        let per = self.sample_size();
        let x = match &self.x {
            Data::F32(v) => Data::F32(gather_flat(v, per, idx)),
            Data::I64(v) => Data::I64(gather_flat(v, per, idx)),
        };
        let y = match &self.y {
            Target::Class(l) => Target::Class(idx.iter().map(|&s| l[s]).collect()),
            Target::Reg { data, shape } => {
                let tp: usize = shape.iter().product();
                Target::Reg { data: gather_flat(data, tp, idx), shape: shape.clone() }
            }
        };
        // rows come from an already-valid dataset -> the shape invariant holds.
        Dataset::build(x, self.sample_shape.clone(), y).expect("gather: sub-dataset invariant")
    }
}
