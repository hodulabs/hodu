//! The runnable-graph artifact: `save_runnable` adds the serialized forward graph and its
//! output/input bindings to the weight rows; `load_runnable` reads them back into a
//! [`RunnableModel`] that runs the forward from the `.hodu` file alone.
use crate::Tensor;
use crate::kurumi::{
    Backend, Feeds, Graph, InputBinding, InputRole, NodeId, Storage, TensorVal, deserialize_graph, serialize_reachable,
};
use crate::nn::Module;
use crate::serialize::container::{DT_U8, Entry, bytes_to_f32, inval, meta, read_container, write_container};
use crate::serialize::model::model_entries;
use std::io;
use std::path::Path;

/// Write a runnable inference artifact: the weight rows (as [`save`](super::save)) PLUS the
/// graph and its output/input bindings, so the model runs from the file alone. `runtime_inputs`
/// names the non-weight Input tensors (x, tokens, ...) fed at inference. Only the nodes the
/// outputs depend on are written, so a training Ctx's backward nodes are pruned; record
/// `outputs` in eval mode so any dropout collapses to identity. Read back with [`load_runnable`].
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
    let out_ids: Vec<NodeId> = outputs.iter().map(|t| t.node()).collect();
    let blob = first.ctx().with_graph(|g| serialize_reachable(g, &out_ids, &inputs));
    write_container(path, &meta(), &model_entries(model), &blob)
}

/// A `.hodu` runnable artifact loaded back into memory: the rebuilt forward graph, its weight
/// feeds already resolved from the file's tensor rows, and the runtime inputs the caller still
/// supplies. Produced by [`load_runnable`]; evaluate with [`RunnableModel::run`].
pub struct RunnableModel {
    graph: Graph,
    outputs: Vec<NodeId>,
    // weight Inputs bound by name from the container rows (node -> value), fed on every run.
    weights: Vec<(NodeId, TensorVal)>,
    // runtime Inputs the caller feeds by name (name -> node); shape comes from the graph node.
    runtime: Vec<(String, NodeId)>,
}

impl RunnableModel {
    /// The names of the runtime inputs this artifact expects (the non-weight leaves, e.g. `"x"`).
    pub fn input_names(&self) -> Vec<&str> {
        self.runtime.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Evaluate every output on `backend`, feeding the stored weights plus the caller's runtime
    /// inputs (each an f32 slice sized to its graph Input node). Errors if a required runtime
    /// input is missing.
    pub fn run(&self, backend: &dyn Backend, runtime: &[(&str, &[f32])]) -> io::Result<Vec<TensorVal>> {
        let mut feeds = Feeds::new();
        for (node, val) in &self.weights {
            feeds.insert(*node, val.clone());
        }
        for (name, node) in &self.runtime {
            let data = runtime
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .ok_or_else(|| inval(format!("run: missing runtime input '{name}'")))?
                .1;
            let shape = self.graph.shape(*node).to_vec();
            feeds.insert(*node, TensorVal { shape, storage: Storage::F32(data.to_vec()) });
        }
        Ok(backend.eval_many_with(&self.graph, &self.outputs, &feeds))
    }
}

/// Load a runnable artifact written by [`save_runnable`]: the weight rows plus the trailing
/// forward-graph blob. Weights are resolved by name against the rows here, so the returned
/// [`RunnableModel`] just needs the runtime inputs at [`run`](RunnableModel::run) time. Errors on
/// a non-runnable file (no graph section), a malformed blob, or a weight the rows are missing.
pub fn load_runnable(path: impl AsRef<Path>) -> io::Result<RunnableModel> {
    let (entries, blob) = read_container(path)?;
    if blob.is_empty() {
        return Err(inval("load_runnable: .hodu has no graph section (not a runnable artifact)"));
    }
    let r = deserialize_graph(&blob).map_err(|e| inval(format!("load_runnable: {e:?}")))?;
    let mut weights = Vec::new();
    let mut runtime = Vec::new();
    for b in &r.inputs {
        match b.role {
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

// A container tensor row -> a feedable value: f32 params/buffers, or a raw-u8 quant byte-buffer.
fn row_to_val(e: &Entry) -> TensorVal {
    let storage = if e.dtype == DT_U8 { Storage::U8(e.data.clone()) } else { Storage::F32(bytes_to_f32(&e.data)) };
    TensorVal { shape: e.shape.clone(), storage }
}

#[cfg(test)]
mod tests;
