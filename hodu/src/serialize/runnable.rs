//! The runnable-graph artifact: `save_runnable` adds the serialized forward graph and its
//! output/input bindings to the weight rows; `load_runnable` reads them back into a
//! [`RunnableModel`] that runs the forward from the `.hodu` file alone.
use crate::Tensor;
use crate::kurumi::{
    Backend, DType, Feeds, Graph, InputBinding, InputRole, NodeId, Storage, TensorVal, deserialize_graph,
    serialize_multi, serialize_reachable,
};
use crate::nn::Module;
use crate::serialize::container::{DT_U8, Entry, bytes_to_f32, inval, meta, read_container, write_container};
use crate::serialize::model::model_entries;
use std::io;
use std::path::Path;

/// Write a runnable inference artifact: the weight rows (as [`save`](super::save)) PLUS the
/// graph and its output/input bindings, so the model runs from the file alone. `runtime_inputs`
/// names the non-weight Input tensors (x, tokens, ...) fed at inference. Only the nodes the
/// outputs depend on are written, so a training Ctx's backward nodes are pruned. The internal
/// RNG Inputs (train/eval flag, dropout seeds) are bound too, so the artifact is self-contained:
/// [`load_runnable`] auto-feeds their eval-mode defaults and the caller feeds only the real
/// runtime inputs. Read back with [`load_runnable`].
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
/// cone is written. [`load_runnable`] loads entry 0 (list the forward/inference entry first).
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

/// A `.hodu` runnable artifact loaded back into memory: the rebuilt forward graph, its weight
/// feeds already resolved from the file's tensor rows, and the runtime inputs the caller still
/// supplies. Produced by [`load_runnable`]; evaluate with [`RunnableModel::run`].
pub struct RunnableModel {
    graph: Graph,
    outputs: Vec<NodeId>,
    // Fixed Inputs fed on every run (node -> value): weights resolved from the container rows,
    // plus internal RNG Inputs auto-set to their eval-mode default (train flag 0.0, seeds 0).
    weights: Vec<(NodeId, TensorVal)>,
    // runtime Inputs the caller feeds by name (name -> node); shape comes from the graph node.
    runtime: Vec<(String, NodeId)>,
}

impl RunnableModel {
    /// The names of the runtime inputs this artifact expects (the non-weight leaves, e.g. `"x"`).
    pub fn input_names(&self) -> Vec<&str> {
        self.runtime.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Evaluate every output on `backend`, feeding the stored weights (and internal eval-mode
    /// RNG defaults) plus the caller's runtime inputs. Each runtime input is a typed [`Storage`]
    /// (f32 data, i64 tokens, ...) sized to its graph Input node. Errors if a required runtime
    /// input is missing.
    pub fn run(&self, backend: &dyn Backend, runtime: &[(&str, Storage)]) -> io::Result<Vec<TensorVal>> {
        let mut feeds = Feeds::new();
        for (node, val) in &self.weights {
            feeds.insert(*node, val.clone());
        }
        for (name, node) in &self.runtime {
            let storage = runtime
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .ok_or_else(|| inval(format!("run: missing runtime input '{name}'")))?
                .1
                .clone();
            let shape = self.graph.shape(*node);
            feeds.insert(*node, TensorVal { shape, storage });
        }
        Ok(backend.eval_many_with(&self.graph, &self.outputs, &feeds))
    }
}

/// Load a runnable artifact written by [`save_runnable`]: the weight rows plus the trailing
/// forward-graph blob. Weights are resolved by name against the rows here, so the returned
/// [`RunnableModel`] just needs the runtime inputs at [`run`](RunnableModel::run) time. Errors on
/// a non-runnable file (no graph section), a malformed blob, or a weight the rows are missing.
pub fn load_runnable(path: impl AsRef<Path>) -> io::Result<RunnableModel> {
    // The quant-descriptor table is informational for a runnable (weights are bound by name from
    // the rows / baked into the graph), so it is read and ignored here.
    let (entries, _descriptors, blob) = read_container(path)?;
    if blob.is_empty() {
        return Err(inval("load_runnable: .hodu has no graph section (not a runnable artifact)"));
    }
    let r = deserialize_graph(&blob).map_err(|e| inval(format!("load_runnable: {e:?}")))?;
    let mut weights = Vec::new();
    let mut runtime = Vec::new();
    for b in &r.inputs {
        match b.role {
            // An internal RNG Input (train flag / dropout seed): no stored row -- its
            // eval-mode value is a known constant, synthesized from the node's dtype+shape.
            InputRole::Weight if b.name.starts_with(RNG_MARK) => {
                weights.push((b.node, eval_default(&r.graph, b.node)));
            }
            InputRole::Weight => {
                let e = entries.iter().find(|e| e.name == b.name).ok_or_else(|| {
                    inval(format!("load_runnable: weight '{}' is missing from the .hodu rows", b.name))
                })?;
                weights.push((b.node, row_to_val(e)));
            }
            InputRole::Runtime => runtime.push((b.name.clone(), b.node)),
        }
    }
    Ok(RunnableModel { graph: r.graph, outputs: r.outputs, weights, runtime })
}

// Reserved input-binding name prefix for the internal RNG Inputs. Starts with NUL so it can
// never collide with a module FQN (dot-joined identifiers), letting load tell an auto-fed
// eval default apart from a real weight row.
const RNG_MARK: char = '\0';

// The eval-mode constant fed to an internal RNG Input: all-zeros of its dtype+shape. flag 0.0
// forces every dropout mask all-ones (threshold train_flag*p = 0), which makes the seed value
// irrelevant, so 0 is fine. See nn/dropout.rs and ctx/train.rs.
fn eval_default(g: &Graph, node: NodeId) -> TensorVal {
    let shape = g.shape(node);
    let n: usize = shape.iter().product();
    let storage = if g.dtype(node) == DType::F32 { Storage::F32(vec![0.0; n]) } else { Storage::I64(vec![0; n]) };
    TensorVal { shape, storage }
}

// A container tensor row -> a feedable value: f32 params/buffers, or a raw-u8 quant byte-buffer.
fn row_to_val(e: &Entry) -> TensorVal {
    let storage = if e.dtype == DT_U8 { Storage::U8(e.data.clone()) } else { Storage::F32(bytes_to_f32(&e.data)) };
    TensorVal { shape: e.shape.clone(), storage }
}

#[cfg(test)]
mod tests;
