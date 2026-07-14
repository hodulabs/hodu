//! GGUF (llama.cpp) weight import: read a `.gguf` file's metadata + tensors into named
//! kurumi f32 tensors (`load_gguf`) or warm-start a live `Module` from one by FQN
//! (`apply_gguf`). A GGUF file is WEIGHTS (like `.safetensors`), not a runnable graph:
//! quantized tensors are dequantized to f32 on load (see `dequant`), so an int4/int8
//! checkpoint warm-starts an f32 model. Hand-rolled parser, no crate dependency;
//! little-endian throughout. K-quants Err by name -- they are not decoded yet.
//!
//! Demo: `apply_gguf(&model, "model.gguf", |n| gg_to_fqn[n].clone())` maps each GGUF
//! tensor name to the model's FQN and sets the (dequantized) weights in place.
mod dequant;

use crate::kurumi::{Storage, TensorVal};
use crate::nn::Module;
use crate::serialize::container::inval;
use std::io;
use std::path::Path;

const MAGIC: u32 = 0x4655_4747; // "GGUF" as a little-endian u32

/// `(metadata, tensors)` as returned by [`load_gguf`].
pub type Loaded = (Vec<(String, GgufValue)>, Vec<(String, TensorVal)>);

/// A GGUF metadata value. Arrays never nest (GGUF guarantees a flat element type).
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

// A bounds-checked little-endian cursor over the file bytes.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len());
        let end = end.ok_or_else(|| inval("gguf: unexpected end of file"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    // gguf_string: u64 length then that many UTF-8 bytes.
    fn str(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        let bytes = self.take(n)?.to_vec();
        String::from_utf8(bytes).map_err(|e| inval(format!("gguf: invalid utf8 string: {e}")))
    }
}

// Read one metadata value of the given type tag. count-driven allocations are bounded by
// EOF: each element consumes >=1 byte, so a bogus count Errs in `take`, it can't preallocate huge.
fn read_value(c: &mut Cur, ty: u32) -> io::Result<GgufValue> {
    Ok(match ty {
        0 => GgufValue::U8(c.take(1)?[0]),
        1 => GgufValue::I8(c.take(1)?[0] as i8),
        2 => GgufValue::U16(u16::from_le_bytes(c.take(2)?.try_into().unwrap())),
        3 => GgufValue::I16(i16::from_le_bytes(c.take(2)?.try_into().unwrap())),
        4 => GgufValue::U32(c.u32()?),
        5 => GgufValue::I32(c.u32()? as i32),
        6 => GgufValue::F32(f32::from_le_bytes(c.take(4)?.try_into().unwrap())),
        7 => GgufValue::Bool(c.take(1)?[0] != 0),
        8 => GgufValue::String(c.str()?),
        9 => {
            let elem_ty = c.u32()?;
            if elem_ty == 9 {
                return Err(inval("gguf: nested arrays are not allowed"));
            }
            let count = c.u64()? as usize;
            let mut v = Vec::new();
            for _ in 0..count {
                v.push(read_value(c, elem_ty)?);
            }
            GgufValue::Array(v)
        }
        10 => GgufValue::U64(c.u64()?),
        11 => GgufValue::I64(c.u64()? as i64),
        12 => GgufValue::F64(f64::from_le_bytes(c.take(8)?.try_into().unwrap())),
        other => return Err(inval(format!("gguf: unknown metadata value type {other}"))),
    })
}

// A parsed tensor directory entry. `dims` are already REVERSED to row-major kurumi order.
struct TensorInfo {
    name: String,
    dims: Vec<usize>,
    ggml_type: u32,
    offset: u64, // relative to the tensor-data section start
}

// Parse the header, metadata, and tensor directory; return them plus the byte offset where
// the (aligned) tensor-data section starts.
type Parsed = (Vec<(String, GgufValue)>, Vec<TensorInfo>, usize);
fn parse(bytes: &[u8]) -> io::Result<Parsed> {
    let mut c = Cur { b: bytes, p: 0 };
    if c.u32()? != MAGIC {
        return Err(inval("gguf: bad magic (not a GGUF file)"));
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(inval(format!("gguf: unsupported version {version} (expected 2 or 3)")));
    }
    let tensor_count = c.u64()? as usize;
    let kv_count = c.u64()? as usize;

    let mut meta = Vec::new();
    for _ in 0..kv_count {
        let key = c.str()?;
        let ty = c.u32()?;
        meta.push((key, read_value(&mut c, ty)?));
    }

    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = c.str()?;
        let n_dims = c.u32()? as usize;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(c.u64()? as usize);
        }
        dims.reverse(); // gguf dims are fastest-varying first -> row-major
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        infos.push(TensorInfo { name, dims, ggml_type, offset });
    }

    // Data section starts at the next multiple of `general.alignment` (u32, default 32).
    let align = meta
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| if let GgufValue::U32(a) = v { Some(*a as usize) } else { None })
        .unwrap_or(32)
        .max(1);
    Ok((meta, infos, c.p.div_ceil(align) * align))
}

/// Read a `.gguf` file as `(metadata, tensors)`: every tensor is dequantized to f32 with its
/// dims reversed to row-major kurumi shape. Errs on bad magic/version or an undecodable type.
pub fn load_gguf(path: impl AsRef<Path>) -> io::Result<Loaded> {
    let bytes = std::fs::read(path)?;
    let (meta, infos, data_start) = parse(&bytes)?;
    let data = bytes.get(data_start..).ok_or_else(|| inval("gguf: tensor-data section is past EOF"))?;

    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos {
        let n: usize = info.dims.iter().product();
        let start = info.offset as usize;
        let raw = data.get(start..).ok_or_else(|| inval(format!("gguf: tensor '{}' offset past EOF", info.name)))?;
        let f32v = dequant::dequant(info.ggml_type, raw, n)?;
        tensors.push((info.name, TensorVal { shape: info.dims, storage: Storage::F32(f32v) }));
    }
    Ok((meta, tensors))
}

// Find the file tensor for a model FQN, validate its shape, mark it consumed. O(n^2) linear
// scan -- fine at model tensor counts (mirrors apply_safetensors).
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
    Err(inval(format!("model tensor '{fqn}' is missing from the gguf file")))
}

// load_gguf always yields F32 storage; unreachable otherwise.
fn f32_of(s: &Storage) -> Vec<f32> {
    match s {
        Storage::F32(v) => v.clone(),
        _ => unreachable!("gguf tensors are dequantized to f32"),
    }
}

/// Warm-start `model` from a `.gguf` file: set each param/buffer/byte-buffer by FQN.
/// `name_map` maps a GGUF tensor name to the model's FQN (identity is `|s| s.to_string()`).
/// Strict like `apply_safetensors`: a missing model tensor, a shape mismatch, or a file tensor
/// that maps to nothing all Err. Params/buffers get the f32 values; byte-buffers take the LE bytes.
pub fn apply_gguf(model: &dyn Module, path: impl AsRef<Path>, name_map: impl Fn(&str) -> String) -> io::Result<()> {
    let entries: Vec<(String, TensorVal)> = load_gguf(path)?.1.into_iter().map(|(n, v)| (name_map(&n), v)).collect();
    let mut used = vec![false; entries.len()];

    for (name, p) in model.named_parameters("") {
        p.set(f32_of(find(&entries, &mut used, &name, p.shape())?));
    }
    for (name, b) in model.named_buffers("") {
        b.set(f32_of(find(&entries, &mut used, &name, b.shape())?));
    }
    for (name, b) in model.named_byte_buffers("") {
        b.set_bytes(find(&entries, &mut used, &name, b.shape())?.to_bytes());
    }
    for (i, (name, _)) in entries.iter().enumerate() {
        if !used[i] {
            return Err(inval(format!("gguf tensor mapped to '{name}' has no match in the model")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use hodu_core::Ctx;

    // A minimal little-endian GGUF writer, just enough to exercise the reader.
    #[derive(Default)]
    struct Builder {
        b: Vec<u8>,
    }
    impl Builder {
        fn u32(&mut self, x: u32) {
            self.b.extend(x.to_le_bytes());
        }
        fn u64(&mut self, x: u64) {
            self.b.extend(x.to_le_bytes());
        }
        fn str(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.b.extend(s.as_bytes());
        }
        fn kv_f32(&mut self, key: &str, x: f32) {
            self.str(key);
            self.u32(6);
            self.b.extend(x.to_le_bytes());
        }
        fn kv_str(&mut self, key: &str, val: &str) {
            self.str(key);
            self.u32(8);
            self.str(val);
        }
        // gguf dims are fastest-first, so pass `row_major` and store it reversed.
        fn tensor_info(&mut self, name: &str, row_major: &[usize], ggml_type: u32, offset: u64) {
            self.str(name);
            self.u32(row_major.len() as u32);
            for &d in row_major.iter().rev() {
                self.u64(d as u64);
            }
            self.u32(ggml_type);
            self.u64(offset);
        }
        // Pad to a 32-byte boundary and append the tensor-data blob.
        fn finish(mut self, data: &[u8]) -> Vec<u8> {
            while !self.b.len().is_multiple_of(32) {
                self.b.push(0);
            }
            self.b.extend(data);
            self.b
        }
    }

    // Header + one f32 KV + one string KV; an F32 tensor at data offset 0 and a Q8_0 tensor at 32.
    fn sample() -> Vec<u8> {
        let mut bld = Builder::default();
        bld.u32(MAGIC);
        bld.u32(3); // version
        bld.u64(2); // tensor_count
        bld.u64(2); // metadata_kv_count
        bld.kv_f32("answer", 42.0);
        bld.kv_str("general.name", "hodu-test");
        bld.tensor_info("mat", &[2, 3], 0, 0); // F32, row-major [2,3]
        bld.tensor_info("vec", &[32], 8, 32); // Q8_0, one block, at offset 32

        let mut data = Vec::new();
        for i in 0..6u32 {
            data.extend((i as f32).to_le_bytes()); // 24 bytes of f32 [0..6]
        }
        while data.len() < 32 {
            data.push(0); // pad up to the Q8_0 offset
        }
        data.extend([0x00, 0x3c]); // Q8_0 scale d = 1.0
        data.extend((1..=32i8).map(|q| q as u8)); // qs = [1..32]
        bld.finish(&data)
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn load_parses_metadata_and_dequantizes() {
        let path = write_temp("hodu_gguf_load.gguf", &sample());
        let (meta, tensors) = load_gguf(&path).unwrap();

        assert_eq!(meta[0], ("answer".to_string(), GgufValue::F32(42.0)));
        assert_eq!(meta[1], ("general.name".to_string(), GgufValue::String("hodu-test".to_string())));

        let by_name: std::collections::HashMap<_, _> = tensors.into_iter().collect();
        let mat = &by_name["mat"];
        assert_eq!(mat.shape, vec![2, 3]); // dims reversed to row-major
        assert_eq!(mat.f32(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let vec = &by_name["vec"];
        assert_eq!(vec.shape, vec![32]);
        assert_eq!(vec.f32(), &(1..=32).map(|i| i as f32).collect::<Vec<_>>());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_magic_errs() {
        let bad = vec![0u8; 32];
        let path = write_temp("hodu_gguf_bad.gguf", &bad);
        assert!(load_gguf(&path).is_err(), "a non-GGUF file must Err");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn apply_warm_starts_linear_by_fqn() {
        // A Linear(in=2,out=3): FQN "0" is the weight (shape [2,3]), "1" the bias (shape [3]).
        let mut bld = Builder::default();
        bld.u32(MAGIC);
        bld.u32(3);
        bld.u64(2); // 2 tensors
        bld.u64(0); // no metadata
        bld.tensor_info("0", &[2, 3], 0, 0); // weight F32
        bld.tensor_info("1", &[3], 0, 32); // bias F32

        let w: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
        let b = vec![1.0f32, 2.0, 3.0];
        let mut data = Vec::new();
        for x in &w {
            data.extend(x.to_le_bytes());
        }
        while data.len() < 32 {
            data.push(0);
        }
        for x in &b {
            data.extend(x.to_le_bytes());
        }
        let path = write_temp("hodu_gguf_apply.gguf", &bld.finish(&data));

        let ctx = Ctx::cpu();
        let lin = Linear::new(&ctx, 2, 3, 0);
        apply_gguf(&lin, &path, |s| s.to_string()).unwrap();
        assert_eq!(lin.weight().value(), w, "weight must warm-start from the gguf tensor");
        assert_eq!(lin.bias().unwrap().value(), b, "bias must warm-start from the gguf tensor");
        std::fs::remove_file(&path).ok();
    }
}
