//! Shape movement on `Tensor`: reshape/transpose/permute/flatten and the
//! slice/split/squeeze views -- thin wraps of kurumi's movement ops (each a strided
//! view, so autodiff comes for free). The per-timestep RNN unroll lives on these:
//! `slice` a timestep, `squeeze` the length-1 time axis, `split` a fused gate
//! projection into its four/three gates.
use kurumi::Error;

use crate::Tensor;

impl Tensor {
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.reshape(n, shape))
    }
    pub fn transpose(&self, i: usize, j: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.transpose(n, i, j))
    }
    /// Reorder axes by `perm` (a permutation of `0..rank`); e.g. `[0,2,1,3]` swaps the
    /// middle two axes (the split-heads transpose in attention).
    pub fn permute(&self, perm: Vec<usize>) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.permute(n, perm))
    }

    /// Flatten dims `[start_dim ..]` into one, keeping the leading dims: for
    /// `[N, C, H, W]` and `start_dim = 1` -> `[N, C*H*W]` (conv features -> Linear).
    pub fn flatten(&self, start_dim: usize) -> Result<Tensor, Error> {
        let sh = self.shape();
        let mut out: Vec<usize> = sh[..start_dim].to_vec();
        out.push(sh[start_dim..].iter().product());
        self.reshape(out)
    }

    /// Slice each axis to `[start, end)` (step 1); `ranges` has one pair per axis.
    /// Pull timestep `t` from `[B,T,in]` with `[(0,B),(t,t+1),(0,in)]` -> `[B,1,in]`.
    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.slice(n, ranges))
    }

    /// Split along `axis` into contiguous chunks of `sizes` (which must sum to the
    /// axis length). Splits a fused gate matrix `[B,4H]` into `[i,f,g,o]`.
    pub fn split(&self, sizes: &[usize], axis: usize) -> Result<Vec<Tensor>, Error> {
        let n = self.node();
        let sizes = sizes.to_vec();
        self.ctx().build_many(|g| g.split(n, &sizes, axis))
    }

    /// Drop `axis` if it has length 1 (else no-op): `[B,1,in]` -> `[B,in]` after a
    /// single-timestep slice.
    pub fn squeeze(&self, axis: usize) -> Result<Tensor, Error> {
        let n = self.node();
        self.ctx().build(|g| g.squeeze(n, axis))
    }
}
