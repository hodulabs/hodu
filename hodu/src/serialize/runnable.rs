//! The runnable-graph artifact writer: save_runnable adds the serialized forward graph and its
//! output/input bindings to the weight rows, so a model runs from the .hodu file alone.
use crate::Tensor;
use crate::kurumi::{InputBinding, InputRole, NodeId, serialize_reachable};
use crate::nn::Module;
use crate::serialize::container::{inval, meta, write_container};
use crate::serialize::model::model_entries;
use std::io;
use std::path::Path;

/// Write a runnable inference artifact: the weight rows (as [`save`](super::save)) PLUS the
/// graph and its output/input bindings, so the model runs from the file alone. `runtime_inputs`
/// names the non-weight Input tensors (x, tokens, ...) fed at inference. Only the nodes the
/// outputs depend on are written, so a training Ctx's backward nodes are pruned; record
/// `outputs` in eval mode so any dropout collapses to identity.
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

#[cfg(test)]
mod tests {
    use super::super::container::read_container;
    use super::super::model::bytes_to_f32;
    use super::*;
    use crate::Ctx;
    use crate::kurumi::{Backend, CpuBackend, Feeds, Storage, TensorVal, deserialize_graph, serialize_graph};
    use crate::nn::Linear;

    // A runnable artifact saved from a TRAINING Ctx must prune the backward nodes and, when
    // loaded and fed its weights (from the rows) plus the runtime input, recompute the exact
    // forward value.
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

        // load: weights from the tensor rows, the graph from the trailing blob.
        let (entries, blob) = read_container(&path).unwrap();
        assert!(!blob.is_empty(), "the runnable artifact must carry a graph section");

        // save_runnable prunes to the forward cone, so the blob is smaller than a whole-arena
        // serialize that would still carry the backward nodes.
        let mut whole: Vec<InputBinding> = lin
            .named_parameters("")
            .into_iter()
            .map(|(name, p)| InputBinding { node: p.tensor().node(), role: InputRole::Weight, name })
            .collect();
        whole.push(InputBinding { node: x.node(), role: InputRole::Runtime, name: "x".into() });
        let whole_blob = ctx.with_graph(|g| serialize_graph(g, &[y.node()], &whole));
        assert!(blob.len() < whole_blob.len(), "backward nodes not pruned ({} vs {})", blob.len(), whole_blob.len());

        let r = deserialize_graph(&blob).unwrap();

        // bind every Input: weights by FQN from the rows, the runtime "x" from our data.
        let mut feeds = Feeds::new();
        for b in &r.inputs {
            match b.role {
                InputRole::Weight => {
                    let e = entries.iter().find(|e| e.name == b.name).expect("weight row present");
                    feeds.insert(
                        b.node,
                        TensorVal { shape: e.shape.clone(), storage: Storage::F32(bytes_to_f32(&e.data)) },
                    );
                }
                InputRole::Runtime => {
                    feeds.insert(b.node, TensorVal { shape: vec![3, 2], storage: Storage::F32(xs.clone()) });
                }
            }
        }
        let got = CpuBackend.eval_with(&r.graph, r.outputs[0], &feeds).f32().to_vec();
        assert_eq!(got, want);

        std::fs::remove_file(&path).ok();
    }
}
