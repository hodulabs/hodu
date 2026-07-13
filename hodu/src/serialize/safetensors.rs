//! HuggingFace `.safetensors` import via the `safetensors` crate: read a file into named
//! kurumi tensors (`load_safetensors`) or warm-start a live `Module` from one by FQN
//! (`apply_safetensors`). The byte format is the public standard, so a real HF checkpoint
//! loads and our writes interoperate cross-frontend with hodu-py. Little-endian throughout,
//! matching kurumi's `Storage::from_bytes`.
//!
//! Demo: `apply_safetensors(&model, "model.safetensors", |n| hf_to_fqn[n].clone())` maps each
//! pretrained tensor name to the model's FQN and sets the weights in place.
use crate::kurumi::{DType, Storage, TensorVal};
use crate::nn::Module;
use crate::serialize::container::inval;
use safetensors::{Dtype, SafeTensors};
use std::io;
use std::path::Path;

// safetensors dtype -> kurumi dtype. The float/int/bool set HF checkpoints ship; the MX
// (F4/F6) and FP8 formats are deferred (they need the quant subsystem, not just a tag).
fn map_dtype(dt: Dtype) -> io::Result<DType> {
    Ok(match dt {
        Dtype::BOOL => DType::BOOL,
        Dtype::U8 => DType::U8,
        Dtype::I8 => DType::I8,
        Dtype::I16 => DType::I16,
        Dtype::U16 => DType::U16,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I32 => DType::I32,
        Dtype::U32 => DType::U32,
        Dtype::F32 => DType::F32,
        Dtype::F64 => DType::F64,
        Dtype::I64 => DType::I64,
        Dtype::U64 => DType::U64,
        other => return Err(inval(format!("safetensors dtype {other:?} unsupported"))),
    })
}

/// Read every tensor from a `.safetensors` file as `(name, TensorVal)`, at its native dtype.
pub fn load_safetensors(path: impl AsRef<Path>) -> io::Result<Vec<(String, TensorVal)>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| inval(format!("safetensors: {e}")))?;
    let mut out = Vec::new();
    for (name, view) in st.tensors() {
        let storage = Storage::from_bytes(map_dtype(view.dtype())?, view.data());
        out.push((name, TensorVal { shape: view.shape().to_vec(), storage }));
    }
    Ok(out)
}

// Any supported storage -> f32: model params/buffers are f32, so f16/bf16 widen and ints cast,
// letting f16/bf16 pretrained weights warm-start an f32 model. F8/complex never reach here
// (map_dtype rejects them at load).
fn to_f32(s: &Storage) -> Vec<f32> {
    match s {
        Storage::F32(v) => v.clone(),
        Storage::F64(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::F16(v) => v.iter().map(|&x| x.to_f32()).collect(),
        Storage::BF16(v) => v.iter().map(|&x| x.to_f32()).collect(),
        Storage::I64(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::I32(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::I16(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::I8(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::U64(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::U32(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::U16(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::U8(v) => v.iter().map(|&x| x as f32).collect(),
        Storage::BOOL(v) => v.iter().map(|&x| u8::from(x) as f32).collect(),
        _ => panic!("apply_safetensors: {:?} has no f32 conversion", s.dtype()),
    }
}

// Find the file tensor for a model FQN, validate its shape, mark it consumed. O(n^2) linear
// scan -- fine at model tensor counts (mirrors container::apply_to_model).
fn find<'a>(
    entries: &'a [(String, TensorVal)],
    used: &mut [bool],
    fqn: &str,
    want: &[usize],
) -> io::Result<&'a Storage> {
    for (i, (name, val)) in entries.iter().enumerate() {
        if !used[i] && name == fqn {
            if val.shape.as_slice() != want {
                return Err(inval(format!("tensor '{fqn}' shape {:?} != model {want:?}", val.shape)));
            }
            used[i] = true;
            return Ok(&val.storage);
        }
    }
    Err(inval(format!("model tensor '{fqn}' is missing from the safetensors file")))
}

/// Warm-start `model` from a `.safetensors` file: set each param/buffer/byte-buffer by FQN.
/// `name_map` maps a safetensors tensor name to the model's FQN (identity is `|s| s.to_string()`).
/// Strict like the `.hodu` loader: a missing model tensor, a shape mismatch, or a file tensor that
/// maps to nothing all Err. Params/buffers are cast to f32; byte-buffers take the raw LE bytes.
pub fn apply_safetensors(
    model: &dyn Module,
    path: impl AsRef<Path>,
    name_map: impl Fn(&str) -> String,
) -> io::Result<()> {
    let entries: Vec<(String, TensorVal)> =
        load_safetensors(path)?.into_iter().map(|(n, v)| (name_map(&n), v)).collect();
    let mut used = vec![false; entries.len()];

    for (name, p) in model.named_parameters("") {
        p.set(to_f32(find(&entries, &mut used, &name, p.shape())?));
    }
    for (name, b) in model.named_buffers("") {
        b.set(to_f32(find(&entries, &mut used, &name, b.shape())?));
    }
    for (name, b) in model.named_byte_buffers("") {
        b.set_bytes(find(&entries, &mut used, &name, b.shape())?.to_bytes());
    }
    for (i, (name, _)) in entries.iter().enumerate() {
        if !used[i] {
            return Err(inval(format!("safetensors tensor mapped to '{name}' has no match in the model")));
        }
    }
    Ok(())
}
