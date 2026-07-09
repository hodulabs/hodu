//! Recurrent layers built by UNROLLING a cell over the time axis -- every gate is
//! a primitive with a VJP, so the whole recurrence autodiffs through
//! `loss.grad(&params)` like any other graph.
//!
//! The graph is static, so the unroll length T is BAKED IN at
//! `forward` time (T = `x.shape()[1]`). One built graph => one fixed T => every
//! fed batch must share that same sequence length (pad/bucket upstream if ragged).
mod gru;
mod lstm;

pub use gru::Gru;
pub use lstm::Lstm;
