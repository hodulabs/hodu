//! hodu: a static Rust ML frontend over the kurumi engine. An ergonomic Tensor,
//! nn layers, losses, and optimizers on kurumi's build-once / feed-per-step graph
//! model (no eager, no dynamic shapes -- that is hodu-py's job; this crate is the
//! static, systems/production path).

pub mod data;
pub mod loss;
pub mod metrics;
pub mod nn;
pub mod optim;
pub mod prelude;
pub mod serialize;

pub use hodu_core::kurumi;
pub use hodu_core::{Ctx, DType, Error, NodeId, Tensor};
