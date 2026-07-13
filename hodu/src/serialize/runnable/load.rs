//! The read path: `load_runnable` reads the graph blob + weight rows back into a
//! [`RunnableModel`] that runs the forward from the `.hodu` file alone.
use super::RNG_MARK;
use crate::kurumi::{Backend, DType, Feeds, Graph, InputRole, NodeId, Storage, TensorVal, deserialize_graph};
use crate::serialize::container::{DT_U8, Entry, bytes_to_f32, inval, read_container};
use std::io;
use std::path::Path;

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

/// Load a runnable artifact written by [`save_runnable`](super::save_runnable): the weight rows
/// plus the trailing forward-graph blob. Weights are resolved by name against the rows here, so
/// the returned [`RunnableModel`] just needs the runtime inputs at [`run`](RunnableModel::run)
/// time. Errors on a non-runnable file (no graph section), a malformed blob, or a weight the rows
/// are missing.
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
