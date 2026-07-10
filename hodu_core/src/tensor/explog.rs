//! Exp / log family on `Tensor`: `sqrt`/`exp`/`square`/`ln`/`powf` -- thin wraps of
//! kurumi's transcendental ops (each autodiffs for free through the engine's VJP).
//! Mirrors the engine's `graph/ops/core/explog.rs`.
use crate::Tensor;
use kurumi::Error;

impl Tensor {
    pub fn sqrt(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.sqrt(n))
    }
    pub fn exp(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.exp(n))
    }
    pub fn square(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.square(n))
    }
    /// Natural log (elementwise).
    pub fn ln(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.ln(n))
    }
    /// Elementwise `self ** p`.
    pub fn powf(&self, p: f32) -> Result<Tensor, Error> {
        let e = self.scalar_like(p);
        self.bin(&e, "pow", |g, a, b| g.pow(a, b))
    }
}
