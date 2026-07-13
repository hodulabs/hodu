//! The write path: `save_runnable` / `save_multi` serialize a model's forward graph and its
//! output/input bindings alongside the weight rows, so the artifact runs from the file alone.
use super::RNG_MARK;
use crate::Tensor;
use crate::kurumi::{InputBinding, InputRole, NodeId, serialize_multi, serialize_reachable};
use crate::nn::Module;
use crate::serialize::container::{inval, meta, write_container};
use crate::serialize::model::model_entries;
use std::io;
use std::path::Path;

/// Write a runnable inference artifact: the weight rows (as [`save`](crate::serialize::save)) PLUS
/// the graph and its output/input bindings, so the model runs from the file alone. `runtime_inputs`
/// names the non-weight Input tensors (x, tokens, ...) fed at inference. Only the nodes the
/// outputs depend on are written, so a training Ctx's backward nodes are pruned. The internal
/// RNG Inputs (train/eval flag, dropout seeds) are bound too, so the artifact is self-contained:
/// [`load_runnable`](super::load_runnable) auto-feeds their eval-mode defaults and the caller feeds
/// only the real runtime inputs. Read back with [`load_runnable`](super::load_runnable).
pub fn save_runnable(
    path: impl AsRef<Path>,
    model: &dyn Module,
    outputs: &[&Tensor],
    runtime_inputs: &[(&str, &Tensor)],
) -> io::Result<()> {
    let Some(first) = outputs.first() else {
        return Err(inval("save_runnable: no outputs given"));
    };
    // Every Input node bound by name: params/buffers are weights (bound from the rows on
    // load), the caller's data tensors are runtime feeds.
    let mut inputs: Vec<InputBinding> = Vec::new();
    for (name, p) in model.named_parameters("") {
        inputs.push(InputBinding { node: p.tensor().node(), role: InputRole::Weight, name });
    }
    for (name, b) in model.named_buffers("") {
        inputs.push(InputBinding { node: b.tensor().node(), role: InputRole::Weight, name });
    }
    for (name, b) in model.named_byte_buffers("") {
        inputs.push(InputBinding { node: b.tensor().node(), role: InputRole::Weight, name });
    }
    for &(name, t) in runtime_inputs {
        inputs.push(InputBinding { node: t.node(), role: InputRole::Runtime, name: name.to_string() });
    }
    // The shared train/eval flag and each dropout seed are graph Inputs no weight/runtime
    // binding covers. Bind them under a reserved-name marker (RNG_MARK) as Weights so load
    // auto-feeds their eval-mode defaults (flag 0.0, seed 0) -- the artifact is self-contained,
    // the caller never passes "train"/"seed". serialize_reachable drops any not reachable from
    // the outputs, so a net without Dropout/BatchNorm binds none.
    for (i, (node, _dt)) in first.ctx().rng_inputs().into_iter().enumerate() {
        inputs.push(InputBinding { node, role: InputRole::Weight, name: format!("{RNG_MARK}{i}") });
    }
    let out_ids: Vec<NodeId> = outputs.iter().map(|t| t.node()).collect();
    let blob = first.ctx().with_graph(|g| serialize_reachable(g, &out_ids, &inputs));
    write_container(path, &meta(), &model_entries(model), &model.quant_descriptors(""), &blob)
}

/// One entry point for [`save_multi`]: its name, output tensors, and (name, tensor) runtime inputs.
pub type EntrySpec<'a> = (&'a str, &'a [&'a Tensor], &'a [(&'a str, &'a Tensor)]);

/// Write a MULTI-ENTRY runnable artifact: the weight rows plus one graph blob carrying N named
/// entry points (e.g. `("forward", ..)` and `("forward_backward", ..)`), all sharing one node
/// arena. The model's weights (params/buffers + internal RNG Inputs) are bound once and shared;
/// each entry adds its own outputs and runtime inputs. Only the union of every entry's output
/// cone is written. [`load_runnable`](super::load_runnable) loads entry 0 (list the
/// forward/inference entry first).
pub fn save_multi(path: impl AsRef<Path>, model: &dyn Module, entries: &[EntrySpec<'_>]) -> io::Result<()> {
    let first = entries.first().and_then(|(_, outs, _)| outs.first());
    let Some(first) = first else {
        return Err(inval("save_multi: no entry outputs given"));
    };
    let ctx = first.ctx();
    // Shared weight bindings (params/buffers/byte-buffers + internal RNG Inputs under RNG_MARK),
    // bound once and reused by every entry. serialize_multi drops any unreachable from an entry.
    let mut weights: Vec<InputBinding> = Vec::new();
    for (name, p) in model.named_parameters("") {
        weights.push(InputBinding { node: p.tensor().node(), role: InputRole::Weight, name });
    }
    for (name, b) in model.named_buffers("") {
        weights.push(InputBinding { node: b.tensor().node(), role: InputRole::Weight, name });
    }
    for (name, b) in model.named_byte_buffers("") {
        weights.push(InputBinding { node: b.tensor().node(), role: InputRole::Weight, name });
    }
    for (i, (node, _dt)) in ctx.rng_inputs().into_iter().enumerate() {
        weights.push(InputBinding { node, role: InputRole::Weight, name: format!("{RNG_MARK}{i}") });
    }
    // Per entry: outputs + (shared weights ++ this entry's runtime inputs).
    let per_entry: Vec<(&str, Vec<NodeId>, Vec<InputBinding>)> = entries
        .iter()
        .map(|&(name, outs, runtime_inputs)| {
            let out_ids: Vec<NodeId> = outs.iter().map(|t| t.node()).collect();
            let mut inputs = weights.clone();
            for &(rn, t) in runtime_inputs {
                inputs.push(InputBinding { node: t.node(), role: InputRole::Runtime, name: rn.to_string() });
            }
            (name, out_ids, inputs)
        })
        .collect();
    let refs: Vec<(&str, &[NodeId], &[InputBinding])> =
        per_entry.iter().map(|(n, o, i)| (*n, o.as_slice(), i.as_slice())).collect();
    let blob = ctx.with_graph(|g| serialize_multi(g, &refs));
    write_container(path, &meta(), &model_entries(model), &model.quant_descriptors(""), &blob)
}
