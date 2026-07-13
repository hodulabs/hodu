//! Convolution layers over kurumi's `conv*` ops: direct and transposed, 1/2/3-D. Each is
//! `[N,C,spatial...] -> [N,O,spatial...]` with a weight and a per-output-channel bias
//! broadcast over the spatial map. Direct weights are `[O,C,K...]`; transposed weights are
//! `[C,O,K...]` (in_ch first).
mod direct;
mod transpose;

pub use direct::{Conv1d, Conv2d, Conv3d};
pub use transpose::{ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
