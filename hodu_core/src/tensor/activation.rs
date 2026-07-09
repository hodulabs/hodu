//! Activations on `Tensor`, thin wraps of kurumi's fused ops (softmax/gelu/... are
//! single fused primitives in the engine, so they lower to one kernel and autodiff
//! through the engine's VJP).
use kurumi::Error;

use crate::Tensor;

impl Tensor {
    pub fn relu(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.relu(n))
    }
    pub fn tanh(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.tanh(n))
    }
    pub fn sigmoid(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.sigmoid(n))
    }
    pub fn gelu(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.gelu(n))
    }
    pub fn silu(&self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.silu(n))
    }

    /// Softmax over `axis`.
    pub fn softmax(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.softmax(n, axis))
    }
    /// Numerically-stable log-softmax over `axis`.
    pub fn log_softmax(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.log_softmax(n, axis))
    }
}
