//! Operator overloading for `Tensor`: `&a + &b` (broadcasting) and `&a + 2.0`
//! (scalar). Static shapes are known at record time, so a mismatch is a bug in
//! the caller's code -- these panic with the record-time error; use the `try_*`
//! methods for the `Result` form.
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::Tensor;

macro_rules! bin_op {
    ($Tr:ident, $m:ident, $try:ident) => {
        impl $Tr<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $m(self, rhs: &Tensor) -> Tensor {
                self.$try(rhs).expect(concat!(stringify!($m), ": shape/dtype mismatch"))
            }
        }
        impl $Tr<f32> for &Tensor {
            type Output = Tensor;
            fn $m(self, rhs: f32) -> Tensor {
                let r = self.scalar_like(rhs);
                self.$try(&r).expect(concat!(stringify!($m), ": scalar op failed"))
            }
        }
    };
}

bin_op!(Add, add, try_add);
bin_op!(Sub, sub, try_sub);
bin_op!(Mul, mul, try_mul);
bin_op!(Div, div, try_div);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let n = self.node();
        self.ctx().build_inf(|g| g.neg(n))
    }
}
