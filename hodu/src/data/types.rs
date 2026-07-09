//! The batch input/target payload variants.

/// A batch of samples, either f32 features or i64 token ids.
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// A batch of targets: integer class labels, or flat f32 regression values with a
/// per-sample `shape`.
pub enum Target {
    Class(Vec<usize>),
    Reg { data: Vec<f32>, shape: Vec<usize> },
}
