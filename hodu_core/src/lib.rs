//! hodu core: a static, NumPy-ergonomic Tensor layer over the kurumi engine.
//! Build a graph once, feed inputs per step (kurumi's native model). Broadcasting
//! and dtype promotion are inserted here, before the engine's strict ops -- that
//! split is the frontend's job. No eager, no dynamic shapes: this is the static
//! surface hodu (the user crate) and, later, the AOT path sit on.
pub use kurumi;

mod ctx;
mod tensor;

pub use ctx::Ctx;
pub use kurumi::{DType, Error, NodeId};
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    #[test]
    fn engine_wired() {
        let mut g = kurumi::Graph::new();
        let a = g.constant(vec![1., 2.], vec![2]);
        let b = g.constant(vec![3., 4.], vec![2]);
        let y = g.add(a, b).unwrap();
        assert_eq!(kurumi::interpret(&g, y).f32(), &[4., 6.]);
    }
}
