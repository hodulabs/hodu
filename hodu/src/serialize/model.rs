//! The model<->rows bridge: a Module's named params/buffers/byte-buffers become tensor-table
//! rows (model_entries), and rows load back into a live model by name (apply_to_model). The
//! on-disk byte format itself lives in container.rs.
use crate::nn::Module;
use crate::serialize::container::{DT_F32, DT_U8, Entry, K_BUFFER, K_OPTIM, K_PARAM, K_QBUFFER, inval};
use std::io;

pub(super) fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn kind_name(k: u8) -> &'static str {
    match k {
        K_PARAM => "param",
        K_BUFFER => "buffer",
        K_OPTIM => "optim",
        K_QBUFFER => "byte-buffer",
        _ => "unknown",
    }
}

// The model's params + buffers (f32) + byte-buffers (raw dtype) as named tensor rows
// (FQN naming, stable per arch).
pub(super) fn model_entries(model: &dyn Module) -> Vec<Entry> {
    let mut out = Vec::new();
    for (name, p) in model.named_parameters("") {
        out.push(Entry {
            name,
            kind: K_PARAM,
            dtype: DT_F32,
            shape: p.shape().to_vec(),
            data: f32_to_bytes(&p.value()),
        });
    }
    for (name, b) in model.named_buffers("") {
        out.push(Entry {
            name,
            kind: K_BUFFER,
            dtype: DT_F32,
            shape: b.shape().to_vec(),
            data: f32_to_bytes(&b.value()),
        });
    }
    for (name, b) in model.named_byte_buffers("") {
        out.push(Entry { name, kind: K_QBUFFER, dtype: DT_U8, shape: b.shape().to_vec(), data: b.bytes() });
    }
    out
}

// Find a model tensor by (kind, name) among the file rows, validate its shape, and
// mark it consumed. O(n^2) linear scan -- fine for model tensor counts; swap for a
// name->index map if a file ever holds thousands of tensors.
fn take(
    entries: &[Entry],
    used: &mut [bool],
    kind: u8,
    name: &str,
    want: &[usize],
    want_dtype: u8,
) -> io::Result<Vec<u8>> {
    for (i, e) in entries.iter().enumerate() {
        if !used[i] && e.kind == kind && e.name == name {
            if e.shape.as_slice() != want {
                return Err(inval(format!("tensor '{name}' shape {:?} != model {want:?}", e.shape)));
            }
            if e.dtype != want_dtype {
                return Err(inval(format!("tensor '{name}' dtype {} != model dtype {want_dtype}", e.dtype)));
            }
            used[i] = true;
            return Ok(e.data.clone());
        }
    }
    Err(inval(format!("model {} '{name}' is missing from the .hodu file", kind_name(kind))))
}

// Populate the live model's params + buffers by name; error on any missing or extra
// (non-optim) tensor. optim rows are left for the caller (load_checkpoint) to apply.
pub(super) fn apply_to_model(entries: &[Entry], model: &dyn Module) -> io::Result<()> {
    let mut used = vec![false; entries.len()];
    for (name, p) in model.named_parameters("") {
        let bytes = take(entries, &mut used, K_PARAM, &name, p.shape(), DT_F32)?;
        p.set(bytes_to_f32(&bytes));
    }
    for (name, b) in model.named_buffers("") {
        let bytes = take(entries, &mut used, K_BUFFER, &name, b.shape(), DT_F32)?;
        b.set(bytes_to_f32(&bytes));
    }
    for (name, b) in model.named_byte_buffers("") {
        let bytes = take(entries, &mut used, K_QBUFFER, &name, b.shape(), DT_U8)?;
        b.set_bytes(bytes);
    }
    for (i, e) in entries.iter().enumerate() {
        if !used[i] && e.kind != K_OPTIM {
            return Err(inval(format!(
                "the .hodu file has {} '{}' with no match in the model",
                kind_name(e.kind),
                e.name
            )));
        }
    }
    Ok(())
}
