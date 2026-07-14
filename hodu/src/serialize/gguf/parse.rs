//! GGUF container parser: file bytes -> (metadata KVs, tensor directory, data-section offset).
//! Hand-rolled, little-endian, no crate dependency. The parent dequantizes the tensors and
//! builds the public result; this file only turns bytes into structure.
use crate::serialize::container::inval;
use std::io;

pub(super) const MAGIC: u32 = 0x4655_4747; // "GGUF" as a little-endian u32

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
pub(super) struct TensorInfo {
    pub(super) name: String,
    pub(super) dims: Vec<usize>,
    pub(super) ggml_type: u32,
    pub(super) offset: u64, // relative to the tensor-data section start
}

// Parse the header, metadata, and tensor directory; return them plus the byte offset where
// the (aligned) tensor-data section starts.
pub(super) type Parsed = (Vec<(String, GgufValue)>, Vec<TensorInfo>, usize);
pub(super) fn parse(bytes: &[u8]) -> io::Result<Parsed> {
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
