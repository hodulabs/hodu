//! Mini-batch loader + a batch that feeds itself into the graph's Inputs. Fixed
//! batch width (the tail partial batch is dropped -- the static graph's shape is
//! baked once).
use crate::data::{Data, Dataset, Target, draw, gather_flat};
use hodu_core::{Ctx, NodeId};

/// One mini-batch. `x` is flat over `x_shape` = `[batch, sample_shape...]`; feed it
/// with [`Batch::feed_x`]. Targets are class labels ([`Batch::y_class`]) or f32
/// regression values ([`Batch::feed_y`]).
pub struct Batch {
    pub x: Data,
    pub x_shape: Vec<usize>,
    pub y: Target,
}

impl Batch {
    /// Feed this batch's input into Input `node` (f32 or i64 by kind).
    pub fn feed_x(&self, ctx: &Ctx, node: NodeId) {
        match &self.x {
            Data::F32(v) => ctx.feed(node, v.clone(), self.x_shape.clone()),
            Data::I64(v) => ctx.feed_i64(node, v.clone(), self.x_shape.clone()),
        }
    }

    /// The class labels (panics if this is a regression batch).
    pub fn y_class(&self) -> &[usize] {
        match &self.y {
            Target::Class(l) => l,
            Target::Reg { .. } => {
                panic!("y_class expects class labels, but this batch is regression")
            }
        }
    }

    /// f32 features (panics if this is a token batch); for tests / manual feeds.
    pub fn x_f32(&self) -> &[f32] {
        match &self.x {
            Data::F32(v) => v,
            Data::I64(_) => panic!("x_f32 expects f32 features, but this batch is i64 tokens"),
        }
    }

    /// Feed a regression target batch into Input `node` (panics on a class batch).
    pub fn feed_y(&self, ctx: &Ctx, node: NodeId, batch: usize) {
        match &self.y {
            Target::Reg { data, shape } => {
                let mut s = vec![batch];
                s.extend_from_slice(shape);
                ctx.feed(node, data.clone(), s);
            }
            Target::Class(_) => {
                panic!("feed_y expects regression targets, but this batch is class labels")
            }
        }
    }
}

/// Yields fixed-size batches. When `shuffle`, each [`DataLoader::batches`] call
/// (one epoch) reshuffles the sample order with a seeded, deterministic stream.
pub struct DataLoader {
    ds: Dataset,
    batch: usize,
    shuffle: bool,
    rng: u64,
}

impl DataLoader {
    pub fn new(ds: Dataset, batch: usize, shuffle: bool, seed: u64) -> DataLoader {
        DataLoader { ds, batch, shuffle, rng: seed ^ 0x2545_F491_4F6C_DD1D }
    }

    /// Number of full batches per epoch (the tail partial batch is dropped).
    pub fn len(&self) -> usize {
        self.ds.len / self.batch
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One epoch's batches. Reshuffles first when `shuffle` is set.
    pub fn batches(&mut self) -> Vec<Batch> {
        let mut idx: Vec<usize> = (0..self.ds.len).collect();
        if self.shuffle {
            // Fisher-Yates over a splitmix64 stream (advances self.rng per epoch).
            for i in (1..idx.len()).rev() {
                let j = (draw(&mut self.rng) >> 33) as usize % (i + 1);
                idx.swap(i, j);
            }
        }
        let per = self.ds.sample_size();
        let mut x_shape = vec![self.batch];
        x_shape.extend_from_slice(&self.ds.sample_shape);
        let mut out = Vec::with_capacity(self.len());
        for chunk in idx.chunks_exact(self.batch) {
            let x = match &self.ds.x {
                Data::F32(v) => Data::F32(gather_flat(v, per, chunk)),
                Data::I64(v) => Data::I64(gather_flat(v, per, chunk)),
            };
            let y = match &self.ds.y {
                Target::Class(l) => Target::Class(chunk.iter().map(|&s| l[s]).collect()),
                Target::Reg { data, shape } => {
                    let tp: usize = shape.iter().product();
                    Target::Reg { data: gather_flat(data, tp, chunk), shape: shape.clone() }
                }
            };
            out.push(Batch { x, x_shape: x_shape.clone(), y });
        }
        out
    }
}
