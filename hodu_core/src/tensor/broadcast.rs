//! The broadcasting / promotion machine behind the strict engine builders: `bin`
//! promotes dtype + broadcasts both operands to the NumPy result shape before the
//! engine's strict op, so `&a + &b` and `x.matmul(&w) + b` behave like NumPy.
//! `coerce`/`broadcast_shape`/`promote` are its pieces.
use crate::Tensor;
use kurumi::{DType, Error, Graph, NodeId};

impl Tensor {
    // record a broadcasting binary op: promote dtype + broadcast both operands to
    // the NumPy result shape, then run the engine's strict builder `f`.
    pub(super) fn bin(
        &self,
        rhs: &Tensor,
        op: &'static str,
        f: impl FnOnce(&mut Graph, NodeId, NodeId) -> Result<NodeId, Error>,
    ) -> Result<Tensor, Error> {
        let shape = broadcast_shape(op, self.shape(), rhs.shape())?;
        let dt = promote(self.dtype(), rhs.dtype());
        let (an, ash, adt) = (self.0.node, self.0.shape.clone(), self.0.dtype);
        let (bn, bsh, bdt) = (rhs.0.node, rhs.0.shape.clone(), rhs.0.dtype);
        self.0.ctx.build(|g| {
            let a = coerce(g, an, &ash, adt, &shape, dt)?;
            let b = coerce(g, bn, &bsh, bdt, &shape, dt)?;
            f(g, a, b)
        })
    }
}

// cast then broadcast `n` into (`shape`, `dt`); each step is a no-op if already so.
fn coerce(
    g: &mut Graph,
    mut n: NodeId,
    cur_shape: &[usize],
    cur_dt: DType,
    shape: &[usize],
    dt: DType,
) -> Result<NodeId, Error> {
    if cur_dt != dt {
        n = g.cast(n, dt);
    }
    if cur_shape != shape {
        n = g.broadcast_to(n, shape.to_vec())?;
    }
    Ok(n)
}

/// NumPy broadcast of two shapes: align trailing dims, each pair must be equal or
/// one of them 1 (which stretches). A missing leading dim counts as 1.
fn broadcast_shape(op: &'static str, a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for i in 0..r {
        // align to the right: a missing leading dim counts as 1.
        let ad = if i + a.len() < r { 1 } else { a[i + a.len() - r] };
        let bd = if i + b.len() < r { 1 } else { b[i + b.len() - r] };
        out[i] = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return Err(Error::Shape { op, msg: format!("cannot broadcast {a:?} and {b:?}") });
        };
    }
    Ok(out)
}

/// Result dtype of a mixed-dtype binary op (NumPy-ish, floats only for now).
fn promote(a: DType, b: DType) -> DType {
    use DType::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        _ => a,
    }
}
