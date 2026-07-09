//! `.hodu` v1: a self-describing, named-tensor container. Header (magic + version +
//! ASCII KV meta) then one flat TENSOR TABLE: per entry a FQN name, a kind tag
//! (param | buffer | optim | byte-buffer), a dtype tag (f32 | u8), rank + dims, and a
//! raw LE payload sized by dtype (f32 = 4B, u8 = 1B). `load` populates the live model
//! BY NAME -- validating each shape/dtype and erroring on any missing / extra /
//! mismatched tensor -- instead of relying on `parameters()` order. Byte-buffers carry
//! a `QuantLinear`'s packed U8 weight at its real (small) size.
//!
//! This persists BatchNorm running stats (buffers), so eval-mode inference is correct
//! after a round-trip; `save_checkpoint` additionally stores optimizer state (moments
//! + step) for training resume. std only, no serde.
//!
//! Next step (hodu-plan/02-artifact-format.md): promote this flat table to
//! section-offset regions with 4K page alignment for mmap. The name/kind/dtype/shape
//! schema here is exactly the table that layout builds on.
use std::io;
use std::path::Path;

use crate::Tensor;
use crate::kurumi::{InputBinding, InputRole, NodeId, serialize_graph};
use crate::nn::Module;
use crate::optim::OptState;

mod container;

use container::{
    DT_F32, Entry, K_OPTIM, apply_to_model, bytes_to_f32, f32_to_bytes, inval, meta, model_entries, read_container,
    write_container,
};

/// Write a model's params + buffers (named, self-describing) to `path`.
pub fn save(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    write_container(path, &meta(), &model_entries(model), &[])
}

/// Load params + buffers from `path` into `model` by name. Errors on bad
/// magic/version, an unknown dtype, or any missing / extra / shape-mismatched tensor.
/// Optimizer rows (from a checkpoint) are ignored -- only the model state is applied.
pub fn load(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    let (entries, _) = read_container(path)?;
    apply_to_model(&entries, model)
}

/// Write model (params + buffers) AND optimizer state (moments + step) so a training
/// run can resume. Load with [`load_checkpoint`]; plain [`load`] still reads the model.
pub fn save_checkpoint(path: impl AsRef<Path>, model: &dyn Module, opt: &dyn OptState) -> io::Result<()> {
    let mut entries = model_entries(model);
    for (name, data) in opt.state_dict() {
        entries.push(Entry { name, kind: K_OPTIM, dtype: DT_F32, shape: vec![data.len()], data: f32_to_bytes(&data) });
    }
    write_container(path, &meta(), &entries, &[])
}

/// Restore model AND optimizer from a checkpoint written by [`save_checkpoint`], so a
/// run resumes with moments + step intact.
pub fn load_checkpoint(path: impl AsRef<Path>, model: &dyn Module, opt: &mut dyn OptState) -> io::Result<()> {
    let (entries, _) = read_container(path)?;
    apply_to_model(&entries, model)?;
    let optim_sd: Vec<(String, Vec<f32>)> =
        entries.iter().filter(|e| e.kind == K_OPTIM).map(|e| (e.name.clone(), bytes_to_f32(&e.data))).collect();
    opt.load_state_dict(&optim_sd).map_err(|e| inval(format!("{e:?}")))
}

/// Write a runnable inference artifact: the weight rows (as [`save`]) PLUS the graph and
/// its output/input bindings, so the model runs from the file alone. `runtime_inputs`
/// names the non-weight Input tensors (x, tokens, ...) fed at inference. Build `outputs`
/// on a forward-only, eval-mode `Ctx` (no grad, no dropout state) so the serialized graph
/// is inference-clean -- the whole record arena is written (dead-node pruning aside).
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
    let blob = first.ctx().with_graph(|g| serialize_graph(g, &out_ids, &inputs));
    write_container(path, &meta(), &model_entries(model), &blob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ctx;
    use crate::kurumi::{Backend, CpuBackend, Feeds, Storage, TensorVal, deserialize_graph};
    use crate::nn::Linear;

    // A forward graph saved as a runnable artifact must, when loaded and fed its weights
    // (from the rows) plus the runtime input, recompute the exact forward value.
    #[test]
    fn save_runnable_round_trips() {
        let ctx = Ctx::cpu();
        let lin = Linear::new(&ctx, 2, 1, 0);
        let x = ctx.input(vec![3, 2]);
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ctx.feed(x.node(), xs.clone(), vec![3, 2]);
        let y = lin.forward(&x).unwrap();
        let want = ctx.eval_f32(y.node());

        let path = std::env::temp_dir().join("hodu_save_runnable_test.hodu");
        save_runnable(&path, &lin, &[&y], &[("x", &x)]).unwrap();

        // load: weights from the tensor rows, the graph from the trailing blob.
        let (entries, blob) = read_container(&path).unwrap();
        assert!(!blob.is_empty(), "the runnable artifact must carry a graph section");
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
