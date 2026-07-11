//! Small eval helpers over realized logits: predicted class + accuracy. Hand-rolled
//! identically across the tests/examples before this; one place now.
use crate::Tensor;

/// Index of the max along `axis` for each position off that axis (row-major), e.g. the
/// predicted class per row for `[N, C]` logits with `axis = 1`. Ties resolve to the last
/// max index (matches `Iterator::max_by`).
pub fn argmax(logits: &Tensor, axis: usize) -> Vec<usize> {
    let shape = logits.shape().to_vec();
    let data = logits.realize();
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let base = o * n * inner + i;
            // >= so equal maxima keep the later index (last-wins, like max_by)
            let best = (1..n).fold(0, |b, k| if data[base + k * inner] >= data[base + b * inner] { k } else { b });
            out.push(best);
        }
    }
    out
}

/// Fraction of rows whose top-logit class matches the integer label. `logits` is
/// `[N, classes]` (row-major); `labels[i]` is the target class of row `i`.
pub fn accuracy(logits: &Tensor, labels: &[usize], classes: usize) -> f32 {
    let numel: usize = logits.shape().iter().product();
    assert_eq!(
        numel,
        labels.len() * classes,
        "accuracy: logits {:?} != {} labels x {classes}",
        logits.shape(),
        labels.len()
    );
    let preds = argmax(logits, logits.rank() - 1);
    let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
    correct as f32 / labels.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ctx;

    #[test]
    fn argmax_and_accuracy() {
        let ctx = Ctx::cpu();
        // rows: [0.1,0.9,0.0] -> 1, [2.0,-1.0,0.5] -> 0
        let logits = ctx.constant(vec![0.1, 0.9, 0.0, 2.0, -1.0, 0.5], vec![2, 3]);
        assert_eq!(argmax(&logits, 1), vec![1, 0]);
        assert_eq!(accuracy(&logits, &[1, 0], 3), 1.0);
        assert_eq!(accuracy(&logits, &[0, 0], 3), 0.5);
    }
}
