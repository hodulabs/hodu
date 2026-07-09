//! Grad post-processing: realize grad tensors to host values, and global-norm clip.
use hodu_core::Tensor;

/// Realize a set of grad tensors to host values (order preserved). All grads come
/// from one `loss.grad(&params)`, so they share a ctx; `eval_many_f32` walks them in
/// a SINGLE shared pass -- the forward+backward trunk common to every grad computes
/// once (no per-grad re-traversal, no cliff for a transformer's ~35 params). Empty in
/// (a param-free model of only activations/dropout/pool) -> empty out, no panic.
pub fn grad_values(grads: &[Tensor]) -> Vec<Vec<f32>> {
    let Some(first) = grads.first() else {
        return Vec::new();
    };
    let ctx = first.ctx();
    let ids: Vec<_> = grads.iter().map(Tensor::node).collect();
    ctx.eval_many_f32(&ids)
}

/// Sum a micro-batch's `grads` into `acc` element-wise (gradient accumulation). An
/// empty `acc` is seeded with a copy of `grads`; later calls add in place. After N
/// micro-batches, `scale_grads(&mut acc, 1.0 / N as f32)` for the mean (matching a
/// single N-wide batch), then one `opt.step(&acc)`.
pub fn accumulate_grads(acc: &mut Vec<Vec<f32>>, grads: &[Vec<f32>]) {
    if acc.is_empty() {
        acc.extend(grads.iter().cloned());
        return;
    }
    assert_eq!(acc.len(), grads.len(), "accumulate_grads: param count differs");
    for (a, g) in acc.iter_mut().zip(grads) {
        for (ai, gi) in a.iter_mut().zip(g) {
            *ai += gi;
        }
    }
}

/// Scale every grad element by `factor` in place -- e.g. `1.0 / n` to average `n`
/// accumulated micro-batches before `opt.step`.
pub fn scale_grads(grads: &mut [Vec<f32>], factor: f32) {
    for g in grads.iter_mut() {
        for x in g.iter_mut() {
            *x *= factor;
        }
    }
}

/// Clip grads in place by global L2 norm: if the norm over ALL grads exceeds
/// `max_norm`, scale every element down to hit it. Returns the pre-clip norm. Call
/// on `grad_values(&grads)` before `opt.step`.
pub fn clip_grad_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let total: f32 = grads.iter().flat_map(|g| g.iter()).map(|x| x * x).sum::<f32>().sqrt();
    if total > max_norm && total > 0.0 {
        let s = max_norm / total;
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= s;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_scales_to_max_norm() {
        let mut g = vec![vec![3.0, 4.0]]; // norm 5
        let pre = clip_grad_norm(&mut g, 1.0);
        assert!((pre - 5.0).abs() < 1e-6);
        let post = (g[0][0].powi(2) + g[0][1].powi(2)).sqrt();
        assert!((post - 1.0).abs() < 1e-6, "post {post}");
        // under the cap -> untouched
        let mut h = vec![vec![0.3, 0.4]];
        clip_grad_norm(&mut h, 1.0);
        assert_eq!(h, vec![vec![0.3, 0.4]]);
    }

    #[test]
    fn grad_values_empty_is_empty() {
        // a param-free model (only activations/dropout/pool) has no grads -> no panic.
        assert!(grad_values(&[]).is_empty());
    }

    #[test]
    fn accumulate_then_average() {
        // Two micro-batch grad sets sum element-wise; scaling by 1/2 gives the mean.
        let mut acc: Vec<Vec<f32>> = Vec::new();
        accumulate_grads(&mut acc, &[vec![1.0, 2.0], vec![3.0]]);
        accumulate_grads(&mut acc, &[vec![3.0, 4.0], vec![5.0]]);
        assert_eq!(acc, vec![vec![4.0, 6.0], vec![8.0]]);
        scale_grads(&mut acc, 0.5);
        assert_eq!(acc, vec![vec![2.0, 3.0], vec![4.0]]);
    }
}
