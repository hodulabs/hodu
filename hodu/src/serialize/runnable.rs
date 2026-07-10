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
mod tests {
    use super::*;
    use crate::Ctx;
    use crate::kurumi::{CpuBackend, serialize_graph};
    use crate::nn::Linear;

    // A runnable artifact saved from a TRAINING Ctx must prune the backward nodes, then load and
    // run from the file alone -- weights from the rows, the runtime input from the caller --
    // recomputing the exact forward value.
    #[test]
    fn save_runnable_round_trips() {
        let ctx = Ctx::cpu();
        let lin = Linear::new(&ctx, 2, 1, 0);
        let x = ctx.input(vec![3, 2]);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ctx.feed(x.node(), xs.clone(), vec![3, 2]);
        let y = lin.forward(&x).unwrap();
        let want = ctx.eval_f32(y.node());

        // a training run: grad() grows the arena with backward nodes the artifact must drop.
        let params = lin.parameters();
        let pts: Vec<&Tensor> = params.iter().map(|p| p.tensor()).collect();
        let _ = y.grad(&pts).unwrap();

        let path = std::env::temp_dir().join("hodu_save_runnable_test.hodu");
        save_runnable(&path, &lin, &[&y], &[("x", &x)]).unwrap();

        // save_runnable prunes to the forward cone, so the blob is smaller than a whole-arena
        // serialize that would still carry the backward nodes.
        let (_, blob) = read_container(&path).unwrap();
        assert!(!blob.is_empty(), "the runnable artifact must carry a graph section");
        let mut whole: Vec<InputBinding> = lin
            .named_parameters("")
            .into_iter()
            .map(|(name, p)| InputBinding { node: p.tensor().node(), role: InputRole::Weight, name })
            .collect();
        whole.push(InputBinding { node: x.node(), role: InputRole::Runtime, name: "x".into() });
        let whole_blob = ctx.with_graph(|g| serialize_graph(g, &[y.node()], &whole));
        assert!(blob.len() < whole_blob.len(), "backward nodes not pruned ({} vs {})", blob.len(), whole_blob.len());

        // load + run through the public API: weights from the rows, "x" from the caller.
        let model = load_runnable(&path).unwrap();
        assert_eq!(model.input_names(), vec!["x"]);
        let got = model.run(&CpuBackend, &[("x", &xs)]).unwrap();
        assert_eq!(got[0].f32(), want.as_slice());

        std::fs::remove_file(&path).ok();
    }

    // Dev tool (run with `--ignored --nocapture`): (re)generate the checked-in cross-frontend
    // fixture that hodu-py loads in tests/test_serialize.py, proving a Rust-written .hodu byte
    // format + graph blob load and run in Python. Prints the input and expected output to
    // hardcode there. Deterministic: Linear seed 0 + fixed x.
    #[test]
    #[ignore = "regenerates the committed hodu-py cross-frontend fixture; run manually"]
    fn gen_cross_frontend_fixture() {
        let ctx = Ctx::cpu();
        let lin = Linear::new(&ctx, 4, 3, 0);
        let x = ctx.input(vec![2, 4]);
        let xs: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 - 0.4).collect();
        ctx.feed(x.node(), xs.clone(), vec![2, 4]);
        let y = lin.forward(&x).unwrap();
        let want = ctx.eval_f32(y.node());

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hodu-py/tests/fixtures/linear.hodu");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        save_runnable(&path, &lin, &[&y], &[("x", &x)]).unwrap();

        println!("wrote {}", path.display());
        println!("x = {xs:?}");
        println!("want = {want:?}");
    }
}
