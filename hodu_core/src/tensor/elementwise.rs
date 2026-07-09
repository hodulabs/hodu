//! Elementwise math on `Tensor`: the broadcasting binary ops (the `try_*` Result
//! forms) and unary maps. Binary ops promote dtype + broadcast shape via `bin`
//! before the engine's strict builders; `&a + &b` operators wrap these.
use kurumi::{DType, Error};

use crate::Tensor;

impl Tensor {
    pub fn try_add(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        self.bin(rhs, "add", |g, a, b| g.add(a, b))
    }
    pub fn try_sub(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        self.bin(rhs, "sub", |g, a, b| g.sub(a, b))
    }
    pub fn try_mul(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        self.bin(rhs, "mul", |g, a, b| g.mul(a, b))
    }
    pub fn try_div(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        self.bin(rhs, "div", |g, a, b| g.div(a, b))
    }
    pub fn max(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        self.bin(rhs, "max", |g, a, b| g.max(a, b))
    }
    pub fn min(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        // min(a,b) = -max(-a,-b): kurumi surfaces max; keep the frontend thin.
        self.bin(rhs, "min", |g, a, b| {
            let na = g.neg(a);
            let nb = g.neg(b);
            let m = g.max(na, nb)?;
            Ok(g.neg(m))
        })
    }

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

    /// Cast to `dtype`. `realize()` reads f32, so `t.cast(DType::F32)` makes an
    /// integer/bool result (e.g. from `argmax`) readable on the host.
    pub fn cast(&self, dtype: DType) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.cast(n, dtype))
    }
    /// Clamp elementwise into `[lo, hi]`.
    pub fn clamp(&self, lo: f32, hi: f32) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.clamp(n, lo, hi))
    }
}
