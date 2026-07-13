//! Convolution layers over kurumi's `conv*` ops: direct and transposed, 1/2/3-D. Each is
//! `[N,C,spatial...] -> [N,O,spatial...]` with a weight and a per-output-channel bias
//! broadcast over the spatial map. Direct weights are `[O,C,K...]`; transposed weights are
//! `[C,O,K...]` (in_ch first). One layer struct per file.
mod conv1d;
mod conv2d;
mod conv3d;
mod transpose1d;
mod transpose2d;
mod transpose3d;

pub use conv1d::Conv1d;
pub use conv2d::Conv2d;
pub use conv3d::Conv3d;
pub use transpose1d::ConvTranspose1d;
pub use transpose2d::ConvTranspose2d;
pub use transpose3d::ConvTranspose3d;
