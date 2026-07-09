//! Neural-net building blocks over the static engine. A `Param` is an Input leaf
//! plus its host value (the optimizer mutates it and re-feeds; the graph node
//! stays fixed). `Module`s compose into a forward graph built once, then trained
//! by feeding batches.
use hodu_core::{Error, Tensor};

mod activation;
mod attention;
mod conv;
mod dropout;
mod embedding;
mod flatten;
mod init;
mod linear;
mod norm;
mod param;
mod pool;
mod quant;
mod rnn;
mod sequential;
mod transformer;

pub use activation::{Gelu, Relu, Sigmoid, Tanh};
pub use attention::MultiHeadAttention;
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use init::{kaiming_normal, normal, uniform, xavier_uniform};
pub use linear::{Init, Linear};
pub use norm::{BatchNorm, BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm, RmsNorm};
pub use param::{Buffer, Param, QBuffer};
pub use pool::{AvgPool2d, MaxPool2d};
pub use quant::QuantLinear;
pub use rnn::{Gru, Lstm};
pub use sequential::Sequential;
pub use transformer::{TransformerBlock, TransformerEncoder};

/// A composable layer: a forward pass plus the tensors it owns. A LEAF (Linear,
/// BatchNorm, ...) overrides `parameters`/`buffers`/`byte_buffers` to report its own
/// tensors; a CONTAINER (Sequential, TransformerBlock, ...) overrides only `children`
/// to name its sub-modules, and every flat/named walk derives from that ONE method --
/// so a child's params, buffers, AND byte-buffers can never be silently dropped from
/// one walk but not another. Object-safe, so `Sequential` can hold `Box<dyn Module>`.
pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error>;

    /// This module's named sub-modules (empty for a leaf). The single source of truth
    /// for recursion: every `parameters`/`buffers`/`byte_buffers`/`named_*` default
    /// walks it. Each name is one FQN path segment (`"0"` for a `Sequential` slot,
    /// `"ln1"`/`"q"` for a field), stable per architecture so `.hodu` files round-trip.
    fn children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// All learnable params, flat and recursive (the optimizer contract). A leaf
    /// overrides to return its own; a container gets the default: its children's, in
    /// order.
    fn parameters(&self) -> Vec<Param> {
        self.children().iter().flat_map(|(_, c)| c.parameters()).collect()
    }

    /// Non-learnable host buffers (e.g. BatchNorm running stats), flat and recursive.
    /// Persisted by `save`/`load` so eval-mode state survives a round-trip. A leaf
    /// overrides to return its own; a container aggregates its children's.
    fn buffers(&self) -> Vec<Buffer> {
        self.children().iter().flat_map(|(_, c)| c.buffers()).collect()
    }

    /// Non-learnable RAW-BYTE buffers of a non-f32 dtype (e.g. a `QuantLinear`'s packed
    /// U8 weight), flat and recursive. Persisted at their true dtype so quantized
    /// weights round-trip. A leaf overrides to return its own; a container aggregates.
    fn byte_buffers(&self) -> Vec<QBuffer> {
        self.children().iter().flat_map(|(_, c)| c.byte_buffers()).collect()
    }

    /// Params keyed by a stable FQN under `prefix` -- the key the self-describing
    /// container loads by (order-independent, unlike `parameters()`). A leaf numbers
    /// its params/buffers/byte-buffers with ONE continuous counter (params first, so a
    /// leaf's params are `{prefix}0..`, its buffers continue where those stop, its
    /// byte-buffers after that) -- every FQN is unique by name alone, not just per
    /// (kind, name); a container recurses each child under `{prefix}{name}.`. Derived
    /// from `children()`, so it can't drift from `named_buffers`/`named_byte_buffers`.
    ///
    /// a module with BOTH own params and children would drop the own ones --
    /// no such hybrid exists here (leaves have no children, containers own nothing
    /// directly); give a container its own tensors via a leaf child if that changes.
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        let children = self.children();
        if children.is_empty() {
            return number(prefix, 0, self.parameters());
        }
        children.iter().flat_map(|(name, c)| c.named_parameters(&format!("{prefix}{name}."))).collect()
    }

    /// Buffers keyed by a stable FQN under `prefix` (see [`Module::named_parameters`]).
    fn named_buffers(&self, prefix: &str) -> Vec<(String, Buffer)> {
        let children = self.children();
        if children.is_empty() {
            return number(prefix, self.parameters().len(), self.buffers());
        }
        children.iter().flat_map(|(name, c)| c.named_buffers(&format!("{prefix}{name}."))).collect()
    }

    /// Byte-buffers keyed by a stable FQN under `prefix` (see [`Module::named_parameters`]).
    fn named_byte_buffers(&self, prefix: &str) -> Vec<(String, QBuffer)> {
        let children = self.children();
        if children.is_empty() {
            return number(prefix, self.parameters().len() + self.buffers().len(), self.byte_buffers());
        }
        children.iter().flat_map(|(name, c)| c.named_byte_buffers(&format!("{prefix}{name}."))).collect()
    }
}

// Number a leaf's own tensors `{prefix}{start}`, `{prefix}{start+1}`, ... `start` is one
// running counter across a leaf's params, then buffers, then byte-buffers, so no two
// FQNs collide by name even across kinds (a name-only index can key on them safely).
fn number<T>(prefix: &str, start: usize, own: Vec<T>) -> Vec<(String, T)> {
    own.into_iter().enumerate().map(|(i, t)| (format!("{prefix}{}", start + i), t)).collect()
}
