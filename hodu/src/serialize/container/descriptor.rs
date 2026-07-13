//! The quant-descriptor table codec: the `[u32 count][descriptors]` section that sits AFTER the
//! tensor table and BEFORE the graph blob (empty -> just a 4-byte zero count). Uses the parent
//! container's shared str/int io helpers.
use super::{inval, read_str, read_u32, read_u64, write_str};
use crate::nn::QuantDescriptor;
use std::io::{self, Read, Write};

// quant-descriptor scheme tags: 0 = affine group-wise (weight-only int4/int8), the only scheme
// today. Reserved: GPTQ/AWQ/MX/codebook get their own tag when added.
const SCHEME_AFFINE: u8 = 0;

pub(super) fn write_descriptors(w: &mut impl Write, descriptors: &[QuantDescriptor]) -> io::Result<()> {
    w.write_all(&(descriptors.len() as u32).to_le_bytes())?;
    for d in descriptors {
        write_str(w, &d.weight_fqn)?;
        w.write_all(&[SCHEME_AFFINE, d.bits])?;
        w.write_all(&(d.group_size as u64).to_le_bytes())?;
        w.write_all(&[d.symmetric as u8])?;
        write_str(w, &d.scales_fqn)?;
        write_str(w, d.mins_fqn.as_deref().unwrap_or(""))?; // empty = none
    }
    Ok(())
}

pub(super) fn read_descriptors(r: &mut impl Read) -> io::Result<Vec<QuantDescriptor>> {
    let nd = read_u32(r)? as usize;
    let mut descriptors = Vec::with_capacity(nd);
    for _ in 0..nd {
        let weight_fqn = read_str(r)?;
        let mut sb = [0u8; 2];
        r.read_exact(&mut sb)?;
        let (scheme, bits) = (sb[0], sb[1]);
        if scheme != SCHEME_AFFINE {
            return Err(inval(format!("quant descriptor '{weight_fqn}': unsupported scheme tag {scheme}")));
        }
        let group_size = read_u64(r)? as usize;
        let mut symb = [0u8; 1];
        r.read_exact(&mut symb)?;
        let scales_fqn = read_str(r)?;
        let mins = read_str(r)?;
        descriptors.push(QuantDescriptor {
            weight_fqn,
            bits,
            group_size,
            symmetric: symb[0] != 0,
            scales_fqn,
            mins_fqn: if mins.is_empty() { None } else { Some(mins) },
        });
    }
    Ok(descriptors)
}
