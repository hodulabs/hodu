//! Parameter-free spatial layers: max and average pooling, 1/2/3-D.
mod avg;
mod max;

pub use avg::{AvgPool1d, AvgPool2d, AvgPool3d};
pub use max::{MaxPool1d, MaxPool2d, MaxPool3d};
