//! Reader for the binary USDC "crate" format (`PXR-USDC`).
//!
//! The byte layouts follow OpenUSD `pxr/usd/sdf/crateFile.cpp` at commit
//! `84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358` (the `dev` branch head when this
//! was written). The functions that implement each step are named after their
//! OpenUSD counterparts in the comments. Versions 0.4.0 through 0.8.x are
//! accepted; older files use layouts this reader does not implement.

use crate::compression::{decode_ints, fast_decompress};
use crate::error::{malformed, Result, UsdError};
use crate::layer::{Spec, SpecKind};
use crate::value::{Arc, ListOp, Specifier, Value, Variability};

const IDENT: &[u8; 8] = b"PXR-USDC";
const BOOTSTRAP_LEN: usize = 88; // ident[8] + version[8] + tocOffset + reserved[8]
const MIN_VERSION: (u8, u8, u8) = (0, 4, 0);
const MAX_VERSION: (u8, u8, u8) = (0, 8, 255);
const MIN_COMPRESSED_ARRAY: usize = 16; // MinCompressedArraySize
const MAX_DEPTH: u32 = 64;
const LIST_END: u32 = u32::MAX; // default-constructed Index terminates a field set

// TypeEnum values from crateDataTypes.h.
const T_BOOL: u8 = 1;
const T_UCHAR: u8 = 2;
const T_INT: u8 = 3;
const T_UINT: u8 = 4;
const T_INT64: u8 = 5;
const T_UINT64: u8 = 6;
const T_HALF: u8 = 7;
const T_FLOAT: u8 = 8;
const T_DOUBLE: u8 = 9;
const T_STRING: u8 = 10;
const T_TOKEN: u8 = 11;
const T_ASSET_PATH: u8 = 12;
const T_MATRIX2D: u8 = 13;
const T_MATRIX3D: u8 = 14;
const T_MATRIX4D: u8 = 15;
const T_QUATD: u8 = 16;
const T_QUATF: u8 = 17;
const T_QUATH: u8 = 18;
const T_VEC2D: u8 = 19;
const T_VEC2F: u8 = 20;
const T_VEC2H: u8 = 21;
const T_VEC2I: u8 = 22;
const T_VEC3D: u8 = 23;
const T_VEC3F: u8 = 24;
const T_VEC3H: u8 = 25;
const T_VEC3I: u8 = 26;
const T_VEC4D: u8 = 27;
const T_VEC4F: u8 = 28;
const T_VEC4H: u8 = 29;
const T_VEC4I: u8 = 30;
const T_DICTIONARY: u8 = 31;
const T_TOKEN_LIST_OP: u8 = 32;
const T_STRING_LIST_OP: u8 = 33;
const T_PATH_LIST_OP: u8 = 34;
const T_REFERENCE_LIST_OP: u8 = 35;
const T_INT_LIST_OP: u8 = 36;
const T_INT64_LIST_OP: u8 = 37;
const T_UINT_LIST_OP: u8 = 38;
const T_UINT64_LIST_OP: u8 = 39;
const T_PATH_VECTOR: u8 = 40;
const T_TOKEN_VECTOR: u8 = 41;
const T_SPECIFIER: u8 = 42;
const T_PERMISSION: u8 = 43;
const T_VARIABILITY: u8 = 44;
const T_VARIANT_SELECTION_MAP: u8 = 45;
const T_TIME_SAMPLES: u8 = 46;
const T_PAYLOAD: u8 = 47;
const T_DOUBLE_VECTOR: u8 = 48;
const T_STRING_VECTOR: u8 = 50;
const T_VALUE_BLOCK: u8 = 51;
const T_PAYLOAD_LIST_OP: u8 = 55;
const T_TIME_CODE: u8 = 56;
const T_DURATION: u8 = 61;
const T_LAYER_OFFSET_VECTOR: u8 = 49;

/// The `TypeEnum` name for an id, as spelled in `crateDataTypes.h`.
pub fn type_name(id: u8) -> &'static str {
    match id {
        1 => "Bool",
        2 => "UChar",
        3 => "Int",
        4 => "UInt",
        5 => "Int64",
        6 => "UInt64",
        7 => "Half",
        8 => "Float",
        9 => "Double",
        10 => "String",
        11 => "Token",
        12 => "AssetPath",
        13 => "Matrix2d",
        14 => "Matrix3d",
        15 => "Matrix4d",
        16 => "Quatd",
        17 => "Quatf",
        18 => "Quath",
        19 => "Vec2d",
        20 => "Vec2f",
        21 => "Vec2h",
        22 => "Vec2i",
        23 => "Vec3d",
        24 => "Vec3f",
        25 => "Vec3h",
        26 => "Vec3i",
        27 => "Vec4d",
        28 => "Vec4f",
        29 => "Vec4h",
        30 => "Vec4i",
        31 => "Dictionary",
        32 => "TokenListOp",
        33 => "StringListOp",
        34 => "PathListOp",
        35 => "ReferenceListOp",
        36 => "IntListOp",
        37 => "Int64ListOp",
        38 => "UIntListOp",
        39 => "UInt64ListOp",
        40 => "PathVector",
        41 => "TokenVector",
        42 => "Specifier",
        43 => "Permission",
        44 => "Variability",
        45 => "VariantSelectionMap",
        46 => "TimeSamples",
        47 => "Payload",
        48 => "DoubleVector",
        49 => "LayerOffsetVector",
        50 => "StringVector",
        51 => "ValueBlock",
        52 => "Value",
        53 => "UnregisteredValue",
        54 => "UnregisteredValueListOp",
        55 => "PayloadListOp",
        56 => "TimeCode",
        57 => "PathExpression",
        58 => "Relocates",
        59 => "Spline",
        60 => "AnimationBlock",
        61 => "Duration",
        _ => "unknown",
    }
}

/// Bounds-checked little-endian reader over a byte slice.
#[derive(Clone)]
struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn at(d: &'a [u8], p: usize) -> Result<Cur<'a>> {
        if p > d.len() {
            return malformed(format!(
                "offset {p} is past the end of the file ({})",
                d.len()
            ));
        }
        Ok(Cur { d, p })
    }

    fn seek(&mut self, p: u64) -> Result<()> {
        if p > self.d.len() as u64 {
            return malformed(format!("seek to {p} past the end of the file"));
        }
        self.p = p as usize;
        Ok(())
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.d.len());
        match end {
            Some(e) => {
                let s = &self.d[self.p..e];
                self.p = e;
                Ok(s)
            }
            None => malformed(format!("read of {n} bytes at {} runs past the end", self.p)),
        }
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let s = self.bytes(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.arr::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.arr()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.arr()?))
    }
}

/// IEEE 754 binary16 to binary32.
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    if exp == 0 {
        sign * frac * 2f32.powi(-24)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// Adds the section name to a structural-table error.
fn in_section<T>(name: &str, r: Result<T>) -> Result<T> {
    r.map_err(|e| match e {
        UsdError::Malformed(m) => UsdError::Malformed(format!("{name} section: {m}")),
        other => other,
    })
}

/// Section names are NUL-padded 15-byte strings.
#[derive(Debug, Clone)]
struct Section {
    name: String,
    start: u64,
    size: u64,
}

/// A parsed crate file with its structural tables decoded.
pub struct CrateFile {
    version: (u8, u8, u8),
    data: Vec<u8>,
    tokens: Vec<String>,
    strings: Vec<u32>,
    /// `(token index, value representation)`
    fields: Vec<(u32, u64)>,
    fieldsets: Vec<u32>,
    paths: Vec<String>,
    /// `(path index, field set index, spec type)`
    specs: Vec<(u32, u32, u32)>,
}

impl std::fmt::Debug for CrateFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrateFile")
            .field("version", &self.version)
            .field("tokens", &self.tokens.len())
            .field("paths", &self.paths.len())
            .field("specs", &self.specs.len())
            .finish()
    }
}

impl CrateFile {
    /// The file's format version, for example `(0, 8, 0)`.
    pub fn version(&self) -> (u8, u8, u8) {
        self.version
    }

    /// Parses the bootstrap, table of contents and all structural sections.
    /// Spec values are decoded later by [`CrateFile::specs`].
    pub fn parse(data: Vec<u8>) -> Result<CrateFile> {
        if data.len() < BOOTSTRAP_LEN {
            return malformed("file is too small to contain the crate bootstrap");
        }
        if &data[..8] != IDENT {
            return malformed("bad crate magic (expected PXR-USDC)");
        }
        let version = (data[8], data[9], data[10]);
        if version < MIN_VERSION || version > MAX_VERSION {
            return Err(UsdError::Unsupported(format!(
                "crate format version {}.{}.{} (this reader handles 0.4.0 to 0.8.x)",
                version.0, version.1, version.2
            )));
        }
        let toc_offset = {
            let mut c = Cur::at(&data, 16)?;
            c.i64()?
        };
        if toc_offset < 0 || toc_offset as usize >= data.len() {
            return malformed("table of contents lies past the end of the file");
        }
        let sections = read_toc(&data, toc_offset as usize)?;
        let section = |name: &str| -> Result<Option<Section>> {
            Ok(sections.iter().find(|s| s.name == name).cloned())
        };

        let mut file = CrateFile {
            version,
            data,
            tokens: Vec::new(),
            strings: Vec::new(),
            fields: Vec::new(),
            fieldsets: Vec::new(),
            paths: Vec::new(),
            specs: Vec::new(),
        };

        // Order mirrors _ReadStructuralSections: tokens first, because every
        // other table refers to them.
        if let Some(s) = section("TOKENS")? {
            file.tokens = in_section("TOKENS", file.read_tokens(&s))?;
        }
        if let Some(s) = section("STRINGS")? {
            let mut c = file.section_cursor(&s)?;
            let n = c.u64()? as usize;
            file.strings = in_section("STRINGS", read_u32s(&mut c, n))?;
        }
        if let Some(s) = section("FIELDS")? {
            file.fields = in_section("FIELDS", file.read_fields(&s))?;
        }
        if let Some(s) = section("FIELDSETS")? {
            file.fieldsets = in_section("FIELDSETS", file.read_fieldsets(&s))?;
        }
        if let Some(s) = section("PATHS")? {
            file.paths = in_section("PATHS", file.read_paths(&s))?;
        }
        if let Some(s) = section("SPECS")? {
            file.specs = in_section("SPECS", file.read_specs(&s))?;
        }
        file.check_tables()?;
        Ok(file)
    }

    fn section_cursor(&self, s: &Section) -> Result<Cur<'_>> {
        Cur::at(&self.data, s.start as usize)
    }

    fn read_tokens(&self, s: &Section) -> Result<Vec<String>> {
        let mut c = self.section_cursor(s)?;
        let n = c.u64()? as usize;
        // Format 0.4.0 and later: sizes, then an LZ4 container of NUL-terminated names.
        let uncompressed = c.u64()? as usize;
        let compressed = c.u64()? as usize;
        let raw = fast_decompress(c.bytes(compressed)?, uncompressed)?;
        let mut out = Vec::with_capacity(n);
        let mut parts = raw.split(|&b| b == 0);
        for _ in 0..n {
            match parts.next() {
                Some(p) => out.push(
                    std::str::from_utf8(p)
                        .map_err(|e| UsdError::Malformed(format!("token is not UTF-8: {e}")))?
                        .to_owned(),
                ),
                None => return malformed(format!("crate claims {n} tokens, found {}", out.len())),
            }
        }
        Ok(out)
    }

    fn read_fields(&self, s: &Section) -> Result<Vec<(u32, u64)>> {
        let mut c = self.section_cursor(s)?;
        let n = c.u64()? as usize;
        let tokens = read_compressed_u32s(&mut c, n)?;
        let reps_size = c.u64()? as usize;
        let reps = fast_decompress(c.bytes(reps_size)?, n * 8)?;
        if reps.len() < n * 8 {
            return malformed("field value representations are truncated");
        }
        Ok(tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&reps[i * 8..i * 8 + 8]);
                (t, u64::from_le_bytes(b))
            })
            .collect())
    }

    fn read_fieldsets(&self, s: &Section) -> Result<Vec<u32>> {
        let mut c = self.section_cursor(s)?;
        let n = c.u64()? as usize;
        let v = read_compressed_u32s(&mut c, n)?;
        if v.last().is_some_and(|&x| x != LIST_END) {
            return malformed("field set table is not terminated");
        }
        Ok(v)
    }

    /// `_ReadCompressedPaths` and `_BuildDecompressedPathsImpl`: the path tree
    /// is stored as three integer arrays, with `jumps` encoding the tree.
    fn read_paths(&self, s: &Section) -> Result<Vec<String>> {
        let mut c = self.section_cursor(s)?;
        // The writer emits the path-table size, then the count of encoded paths
        // (the table minus the empty path), then the three arrays.
        let table = c.u64()? as usize;
        let n = c.u64()? as usize;
        if n > table {
            return malformed("more encoded paths than path table entries");
        }
        let path_indexes = read_compressed_u32s(&mut c, n)?;
        let token_indexes = read_compressed_i32s(&mut c, n)?;
        let jumps = read_compressed_i32s(&mut c, n)?;

        let mut paths = vec![String::new(); table];
        let mut seen = vec![false; table];
        // Each work item is (next index in the arrays, path of the parent).
        let mut work: Vec<(usize, Option<String>)> = vec![(0, None)];
        while let Some((mut cur, mut parent)) = work.pop() {
            loop {
                if cur >= n {
                    return malformed("path tree walks past its table");
                }
                let t = cur;
                cur += 1;
                let pidx = path_indexes[t] as usize;
                if pidx >= table || seen[pidx] {
                    return malformed("path index out of range or repeated");
                }
                seen[pidx] = true;
                let path = match &parent {
                    None => "/".to_owned(),
                    Some(par) => {
                        let tok = token_indexes[t];
                        let name = self.token(tok.unsigned_abs())?;
                        if tok < 0 {
                            format!("{par}.{name}")
                        } else if name.starts_with('{') {
                            // A variant selection element attaches without a slash: /P{set=sel}.
                            format!("{par}{name}")
                        } else if par == "/" {
                            format!("/{name}")
                        } else {
                            format!("{par}/{name}")
                        }
                    }
                };
                paths[pidx] = path.clone();
                let j = jumps[t];
                let has_child = j > 0 || j == -1;
                let has_sibling = j >= 0;
                if has_child && has_sibling {
                    let sib = t as i64 + j as i64;
                    if sib < 0 || sib as usize >= n {
                        return malformed("path jump lands outside the table");
                    }
                    work.push((sib as usize, parent.clone()));
                }
                if !(has_child || has_sibling) {
                    break;
                }
                if has_child {
                    parent = Some(path);
                }
            }
        }
        Ok(paths)
    }

    /// `_ReadCompressedInts` ordering: spec path indexes, field set indexes, types.
    fn read_specs(&self, s: &Section) -> Result<Vec<(u32, u32, u32)>> {
        let mut c = self.section_cursor(s)?;
        let n = c.u64()? as usize;
        let a = read_compressed_u32s(&mut c, n)?;
        let b = read_compressed_u32s(&mut c, n)?;
        let t = read_compressed_u32s(&mut c, n)?;
        Ok((0..n).map(|i| (a[i], b[i], t[i])).collect())
    }

    fn check_tables(&self) -> Result<()> {
        for &(t, _) in &self.fields {
            if t as usize >= self.tokens.len() {
                return malformed("field name index is out of range");
            }
        }
        for &(p, fs, _) in &self.specs {
            if p as usize >= self.paths.len() || fs as usize >= self.fieldsets.len() {
                return malformed("spec refers to a missing path or field set");
            }
        }
        Ok(())
    }

    fn token(&self, idx: u32) -> Result<&str> {
        self.tokens
            .get(idx as usize)
            .map(String::as_str)
            .ok_or_else(|| UsdError::Malformed(format!("token index {idx} out of range")))
    }

    fn string_of(&self, idx: u32) -> Result<String> {
        let tok = *self
            .strings
            .get(idx as usize)
            .ok_or_else(|| UsdError::Malformed(format!("string index {idx} out of range")))?;
        Ok(self.token(tok)?.to_owned())
    }

    fn path_of(&self, idx: u32) -> Result<String> {
        self.paths
            .get(idx as usize)
            .cloned()
            .ok_or_else(|| UsdError::Malformed(format!("path index {idx} out of range")))
    }

    /// Decodes every spec with its fields. Fails on the first field whose
    /// value type this reader does not decode, naming the type id.
    pub fn specs(&self) -> Result<Vec<Spec>> {
        let mut out = Vec::with_capacity(self.specs.len());
        for &(pidx, fsidx, stype) in &self.specs {
            let kind = SpecKind::from_u32(stype).ok_or_else(|| {
                UsdError::Malformed(format!("spec type {stype} is not an SdfSpecType"))
            })?;
            let path = self.path_of(pidx)?;
            let mut fields = Vec::new();
            let mut i = fsidx as usize;
            loop {
                let fi = *self
                    .fieldsets
                    .get(i)
                    .ok_or_else(|| UsdError::Malformed("field set run is not terminated".into()))?;
                if fi == LIST_END {
                    break;
                }
                let (tok, rep) = self.fields[fi as usize];
                let name = self.token(tok)?.to_owned();
                let value = self.unpack(rep, 0).map_err(|e| match e {
                    UsdError::UnsupportedValueType { .. } | UsdError::Unsupported(_) => {
                        UsdError::Unsupported(format!("field {name} of {path}: {e}"))
                    }
                    other => other,
                })?;
                fields.push((name, value));
                i += 1;
            }
            out.push(Spec { path, kind, fields });
        }
        Ok(out)
    }

    // ----- value representations (ValueRep, CrateFile::_UnpackValue) -----

    fn unpack(&self, rep: u64, depth: u32) -> Result<Value> {
        if depth > MAX_DEPTH {
            return malformed("value nesting is too deep (possible cycle)");
        }
        const ARRAY: u64 = 1 << 63;
        const INLINED: u64 = 1 << 62;
        const COMPRESSED: u64 = 1 << 61;
        const ARRAY_EDIT: u64 = 1 << 60;
        let id = ((rep >> 48) & 0xff) as u8;
        let payload = rep & ((1u64 << 48) - 1);
        let low = payload as u32;
        if id == T_VALUE_BLOCK {
            return Ok(Value::ValueBlock);
        }
        if rep & ARRAY_EDIT != 0 {
            return Err(unsupported_type(id));
        }
        if rep & ARRAY != 0 {
            return self.unpack_array(id, payload, rep & COMPRESSED != 0);
        }
        if always_inlined(id) {
            return self.inline_direct(id, low);
        }
        if rep & INLINED != 0 {
            return self.inline_packed(id, low);
        }
        let mut c = Cur::at(&self.data, payload as usize)?;
        match id {
            T_TIME_SAMPLES => self.time_samples(payload, depth),
            _ => self.read_at(&mut c, id, depth),
        }
    }

    /// Values whose bits live in the 4-byte payload itself.
    fn inline_direct(&self, id: u8, low: u32) -> Result<Value> {
        Ok(match id {
            T_BOOL => Value::Bool(low as u8 != 0),
            T_UCHAR => Value::UChar(low as u8),
            T_INT => Value::Int(low as i32),
            T_UINT => Value::UInt(low),
            T_HALF => Value::Float(half_to_f32(low as u16)),
            T_FLOAT => Value::Float(f32::from_bits(low)),
            T_STRING => Value::String(self.string_of(low)?),
            T_TOKEN => Value::Token(self.token(low)?.to_owned()),
            T_ASSET_PATH => Value::AssetPath(self.token(low)?.to_owned()),
            T_VEC2H => {
                let b = low.to_le_bytes();
                Value::Vec2f([
                    half_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    half_to_f32(u16::from_le_bytes([b[2], b[3]])),
                ])
            }
            T_SPECIFIER => Value::Specifier(specifier(low)?),
            T_VARIABILITY => Value::Variability(variability(low)?),
            _ => return Err(unsupported_type(id)),
        })
    }

    /// Values stored inline as small integers or int8 components
    /// (`_DecodeInline`).
    fn inline_packed(&self, id: u8, low: u32) -> Result<Value> {
        let b = low.to_le_bytes();
        let i8s = |n: usize| -> [i64; 4] {
            let mut o = [0i64; 4];
            for k in 0..n {
                o[k] = b[k] as i8 as i64;
            }
            o
        };
        Ok(match id {
            T_FLOAT | T_DOUBLE => {
                let f = f32::from_bits(low) as f64;
                if id == T_FLOAT {
                    Value::Float(f as f32)
                } else {
                    Value::Double(f)
                }
            }
            T_INT64 => Value::Int64(low as i32 as i64),
            T_UINT64 => Value::UInt64(low as u64),
            T_VEC2D | T_VEC3D | T_VEC4D => {
                let n = vec_dim(id);
                let v = i8s(n);
                match n {
                    2 => Value::Vec2d([v[0] as f64, v[1] as f64]),
                    3 => Value::Vec3d([v[0] as f64, v[1] as f64, v[2] as f64]),
                    _ => Value::Vec4d([v[0] as f64, v[1] as f64, v[2] as f64, v[3] as f64]),
                }
            }
            T_VEC2F | T_VEC3F | T_VEC4F | T_VEC2H | T_VEC3H | T_VEC4H => {
                let n = vec_dim(id);
                let v = i8s(n);
                match n {
                    2 => Value::Vec2f([v[0] as f32, v[1] as f32]),
                    3 => Value::Vec3f([v[0] as f32, v[1] as f32, v[2] as f32]),
                    _ => Value::Vec4f([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                }
            }
            T_VEC2I | T_VEC3I | T_VEC4I => {
                let n = vec_dim(id);
                let v = i8s(n);
                match n {
                    2 => Value::Vec2i([v[0] as i32, v[1] as i32]),
                    3 => Value::Vec3i([v[0] as i32, v[1] as i32, v[2] as i32]),
                    _ => Value::Vec4i([v[0] as i32, v[1] as i32, v[2] as i32, v[3] as i32]),
                }
            }
            T_MATRIX2D | T_MATRIX3D | T_MATRIX4D => {
                let n = match id {
                    T_MATRIX2D => 2,
                    T_MATRIX3D => 3,
                    _ => 4,
                };
                let v = i8s(n);
                let mut m = vec![0.0f64; n * n];
                for k in 0..n {
                    m[k * n + k] = v[k] as f64;
                }
                match n {
                    2 => Value::Matrix2d(Box::new([m[0], m[1], m[2], m[3]])),
                    3 => {
                        let mut a = [0.0; 9];
                        a.copy_from_slice(&m);
                        Value::Matrix3d(Box::new(a))
                    }
                    _ => {
                        let mut a = [0.0; 16];
                        a.copy_from_slice(&m);
                        Value::Matrix4d(Box::new(a))
                    }
                }
            }
            T_DICTIONARY => {
                if low != 0 {
                    return malformed("non-empty inline dictionary");
                }
                Value::Dictionary(Vec::new())
            }
            _ => return Err(unsupported_type(id)),
        })
    }

    /// Reads one non-array value (or container) at the cursor.
    fn read_at(&self, c: &mut Cur<'_>, id: u8, depth: u32) -> Result<Value> {
        match id {
            T_DICTIONARY => self.read_dict(c, depth),
            T_TOKEN_LIST_OP | T_STRING_LIST_OP | T_PATH_LIST_OP | T_INT_LIST_OP
            | T_INT64_LIST_OP | T_UINT_LIST_OP | T_UINT64_LIST_OP | T_REFERENCE_LIST_OP
            | T_PAYLOAD_LIST_OP => self.read_list_op(c, id),
            T_PATH_VECTOR | T_TOKEN_VECTOR | T_STRING_VECTOR | T_DOUBLE_VECTOR => {
                let n = c.u64()? as usize;
                let elem = match id {
                    T_PATH_VECTOR => T_PATH_VECTOR,
                    T_TOKEN_VECTOR => T_TOKEN,
                    T_STRING_VECTOR => T_STRING,
                    _ => T_DOUBLE,
                };
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.read_elem(c, elem)?);
                }
                Ok(Value::Array(v))
            }
            T_VARIANT_SELECTION_MAP => {
                let n = c.u64()? as usize;
                let mut m = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    let k = self.string_of(c.u32()?)?;
                    let v = self.string_of(c.u32()?)?;
                    m.push((k, Value::String(v)));
                }
                Ok(Value::Dictionary(m))
            }
            T_PAYLOAD => Ok(Value::Payload(Box::new(self.read_arc(c, true)?))),
            T_LAYER_OFFSET_VECTOR => {
                let n = c.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    v.push(Value::LayerOffset(c.f64()?, c.f64()?));
                }
                Ok(Value::Array(v))
            }
            T_TIME_SAMPLES => Err(unsupported_type(id)),
            _ => self.read_elem(c, id),
        }
    }

    /// Scalar and struct types read by bitwise copy or by index (`Read<T>`).
    fn read_elem(&self, c: &mut Cur<'_>, id: u8) -> Result<Value> {
        Ok(match id {
            T_BOOL => Value::Bool(c.u8()? != 0),
            T_UCHAR => Value::UChar(c.u8()?),
            T_INT => Value::Int(c.i32()?),
            T_UINT => Value::UInt(c.u32()?),
            T_INT64 => Value::Int64(c.i64()?),
            T_UINT64 => Value::UInt64(c.u64()?),
            T_HALF => Value::Float(half_to_f32(c.u16()?)),
            T_FLOAT => Value::Float(c.f32()?),
            T_DOUBLE => Value::Double(c.f64()?),
            T_STRING => Value::String(self.string_of(c.u32()?)?),
            T_TOKEN => Value::Token(self.token(c.u32()?)?.to_owned()),
            T_ASSET_PATH => Value::AssetPath(self.string_of(c.u32()?)?),
            T_PATH_VECTOR => Value::Path(self.path_of(c.u32()?)?),
            T_MATRIX2D => Value::Matrix2d(Box::new(read_f64s::<4>(c)?)),
            T_MATRIX3D => Value::Matrix3d(Box::new(read_f64s::<9>(c)?)),
            T_MATRIX4D => Value::Matrix4d(Box::new(read_f64s::<16>(c)?)),
            T_QUATD => Value::Quatd(read_f64s::<4>(c)?),
            T_QUATF => Value::Quatf(read_f32s::<4>(c)?),
            T_QUATH => {
                let h = read_f16s::<4>(c)?;
                Value::Quatf(h)
            }
            T_VEC2D => Value::Vec2d(read_f64s::<2>(c)?),
            T_VEC3D => Value::Vec3d(read_f64s::<3>(c)?),
            T_VEC4D => Value::Vec4d(read_f64s::<4>(c)?),
            T_VEC2F => Value::Vec2f(read_f32s::<2>(c)?),
            T_VEC3F => Value::Vec3f(read_f32s::<3>(c)?),
            T_VEC4F => Value::Vec4f(read_f32s::<4>(c)?),
            T_VEC2H => Value::Vec2f(read_f16s::<2>(c)?),
            T_VEC3H => Value::Vec3f(read_f16s::<3>(c)?),
            T_VEC4H => Value::Vec4f(read_f16s::<4>(c)?),
            T_VEC2I => Value::Vec2i(read_i32s::<2>(c)?),
            T_VEC3I => Value::Vec3i(read_i32s::<3>(c)?),
            T_VEC4I => Value::Vec4i(read_i32s::<4>(c)?),
            T_SPECIFIER => Value::Specifier(specifier(c.u32()?)?),
            T_VARIABILITY => Value::Variability(variability(c.u32()?)?),
            T_TIME_CODE | T_DURATION => Value::Double(c.f64()?),
            T_PAYLOAD => Value::Payload(Box::new(self.read_arc(c, true)?)),
            _ => return Err(unsupported_type(id)),
        })
    }

    /// `SdfReference` (not a payload) or `SdfPayload`. Payloads carry a layer
    /// offset only from format 0.8.0 on.
    fn read_arc(&self, c: &mut Cur<'_>, payload: bool) -> Result<Arc> {
        let asset_path = self.string_of(c.u32()?)?;
        let prim_path = self.path_of(c.u32()?)?;
        let layer_offset = if !payload || self.version >= (0, 8, 0) {
            (c.f64()?, c.f64()?)
        } else {
            (0.0, 1.0)
        };
        let custom_data = if payload {
            Vec::new()
        } else {
            self.read_dict_entries(c, 1)?
        };
        Ok(Arc {
            asset_path,
            prim_path,
            layer_offset,
            custom_data,
        })
    }

    fn read_dict(&self, c: &mut Cur<'_>, depth: u32) -> Result<Value> {
        Ok(Value::Dictionary(self.read_dict_entries(c, depth)?))
    }

    fn read_dict_entries(&self, c: &mut Cur<'_>, depth: u32) -> Result<Vec<(String, Value)>> {
        let n = c.u64()? as usize;
        let mut out = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            let key = self.string_of(c.u32()?)?;
            let value = self.read_vt_value(c, depth + 1)?;
            out.push((key, value));
        }
        Ok(out)
    }

    /// `Read<VtValue>`: an int64 offset, relative to its own position, leads to
    /// the value representation. The cursor stays after that representation.
    fn read_vt_value(&self, c: &mut Cur<'_>, depth: u32) -> Result<Value> {
        let start = c.p as i64;
        let off = c.i64()?;
        let target = start.checked_add(off).filter(|&t| t >= 0);
        let target = target.ok_or_else(|| UsdError::Malformed("negative value offset".into()))?;
        c.seek(target as u64)?;
        let rep = c.u64()?;
        self.unpack(rep, depth)
    }

    fn read_list_op(&self, c: &mut Cur<'_>, id: u8) -> Result<Value> {
        let bits = c.u8()?;
        let elem = match id {
            T_TOKEN_LIST_OP => T_TOKEN,
            T_STRING_LIST_OP => T_STRING,
            T_PATH_LIST_OP => T_PATH_VECTOR,
            T_INT_LIST_OP => T_INT,
            T_INT64_LIST_OP => T_INT64,
            T_UINT_LIST_OP => T_UINT,
            T_UINT64_LIST_OP => T_UINT64,
            _ => T_PAYLOAD,
        };
        let is_ref = id == T_REFERENCE_LIST_OP;
        let read_vec = |c: &mut Cur<'_>| -> Result<Vec<Value>> {
            let n = c.u64()? as usize;
            let mut v = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                let item = if is_ref {
                    Value::Reference(Box::new(self.read_arc(c, false)?))
                } else {
                    self.read_elem(c, elem)?
                };
                v.push(item);
            }
            Ok(v)
        };
        let mut op = ListOp {
            is_explicit: bits & 1 != 0,
            ..ListOp::default()
        };
        if bits & (1 << 1) != 0 {
            op.explicit = read_vec(c)?;
        }
        if bits & (1 << 2) != 0 {
            op.added = read_vec(c)?;
        }
        if bits & (1 << 5) != 0 {
            op.prepended = read_vec(c)?;
        }
        if bits & (1 << 6) != 0 {
            op.appended = read_vec(c)?;
        }
        if bits & (1 << 3) != 0 {
            op.deleted = read_vec(c)?;
        }
        if bits & (1 << 4) != 0 {
            op.ordered = read_vec(c)?;
        }
        Ok(Value::ListOp(Box::new(op)))
    }

    /// `_ReadPossiblyCompressedArray` and `_ReadUncompressedArray`.
    fn unpack_array(&self, id: u8, payload: u64, compressed: bool) -> Result<Value> {
        if payload == 0 {
            return Ok(Value::Array(Vec::new())); // an inlined empty array
        }
        let mut c = Cur::at(&self.data, payload as usize)?;
        let n = c.u64()? as usize;
        let is_int = matches!(id, T_INT | T_UINT | T_INT64 | T_UINT64);
        let is_float = matches!(id, T_HALF | T_FLOAT | T_DOUBLE);
        let raw = |c: &mut Cur<'_>| -> Result<Vec<Value>> {
            let mut v = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                v.push(self.read_elem(c, id)?);
            }
            Ok(v)
        };
        if is_int && compressed && n >= MIN_COMPRESSED_ARRAY {
            let width = if matches!(id, T_INT64 | T_UINT64) {
                8
            } else {
                4
            };
            let ints = read_compressed_ints(&mut c, n, width)?;
            return Ok(Value::Array(
                ints.into_iter().map(|x| int_value(id, x)).collect(),
            ));
        }
        if is_float && compressed && n >= MIN_COMPRESSED_ARRAY {
            let code = c.u8()?;
            return match code {
                b'i' => {
                    let ints = read_compressed_ints(&mut c, n, 4)?;
                    Ok(Value::Array(
                        ints.into_iter()
                            .map(|x| float_value(id, x as f64))
                            .collect(),
                    ))
                }
                b't' => {
                    let lut_len = c.u32()? as usize;
                    let mut lut = Vec::with_capacity(lut_len.min(1 << 20));
                    for _ in 0..lut_len {
                        lut.push(self.read_elem(&mut c, id)?);
                    }
                    let idx = read_compressed_ints(&mut c, n, 4)?;
                    let mut out = Vec::with_capacity(n);
                    for k in idx {
                        let v = lut.get(k as usize).ok_or_else(|| {
                            UsdError::Malformed("float lookup index out of range".into())
                        })?;
                        out.push(v.clone());
                    }
                    Ok(Value::Array(out))
                }
                other => malformed(format!("unknown compressed float array code {other:#x}")),
            };
        }
        Ok(Value::Array(raw(&mut c)?))
    }

    /// `TimeSamples`: the payload leads to the times array representation and
    /// to a table of value representations (see `_RecursiveRead`).
    fn time_samples(&self, payload: u64, depth: u32) -> Result<Value> {
        let mut c = Cur::at(&self.data, payload as usize)?;
        let start = c.p as i64;
        let off = c.i64()?;
        c.seek((start + off) as u64)?;
        let times_rep = c.u64()?;
        let start2 = c.p as i64;
        let off2 = c.i64()?;
        c.seek((start2 + off2) as u64)?;
        let n = c.u64()? as usize;
        let times = match self.unpack(times_rep, depth + 1)? {
            Value::Array(v) => v
                .iter()
                .map(|x| {
                    x.as_f64()
                        .ok_or_else(|| UsdError::Malformed("time is not a double".into()))
                })
                .collect::<Result<Vec<f64>>>()?,
            _ => return malformed("time samples have no time array"),
        };
        if times.len() != n {
            return malformed("time sample count differs from its time array");
        }
        let mut out = Vec::with_capacity(n);
        for t in times {
            let rep = c.u64()?;
            out.push((t, self.unpack(rep, depth + 1)?));
        }
        Ok(Value::TimeSamples(out))
    }
}

fn always_inlined(id: u8) -> bool {
    matches!(
        id,
        T_BOOL
            | T_UCHAR
            | T_INT
            | T_UINT
            | T_HALF
            | T_FLOAT
            | T_STRING
            | T_TOKEN
            | T_ASSET_PATH
            | T_VEC2H
            | T_SPECIFIER
            | T_PERMISSION
            | T_VARIABILITY
    )
}

fn vec_dim(id: u8) -> usize {
    match id {
        T_VEC2D | T_VEC2F | T_VEC2H | T_VEC2I => 2,
        T_VEC3D | T_VEC3F | T_VEC3H | T_VEC3I => 3,
        _ => 4,
    }
}

fn int_value(id: u8, x: i64) -> Value {
    match id {
        T_INT => Value::Int(x as i32),
        T_UINT => Value::UInt(x as u32),
        T_INT64 => Value::Int64(x),
        _ => Value::UInt64(x as u64),
    }
}

fn float_value(id: u8, x: f64) -> Value {
    match id {
        T_DOUBLE => Value::Double(x),
        _ => Value::Float(x as f32),
    }
}

fn unsupported_type(id: u8) -> UsdError {
    UsdError::UnsupportedValueType {
        id,
        name: type_name(id),
    }
}

fn specifier(v: u32) -> Result<Specifier> {
    match v {
        0 => Ok(Specifier::Def),
        1 => Ok(Specifier::Over),
        2 => Ok(Specifier::Class),
        _ => malformed(format!("specifier value {v} is not an SdfSpecifier")),
    }
}

fn variability(v: u32) -> Result<Variability> {
    match v {
        0 => Ok(Variability::Varying),
        // 2 is the legacy "config" variability, which OpenUSD reads as uniform.
        1 | 2 => Ok(Variability::Uniform),
        _ => malformed(format!("variability value {v} is not an SdfVariability")),
    }
}

fn read_u32s(c: &mut Cur<'_>, n: usize) -> Result<Vec<u32>> {
    let mut v = Vec::with_capacity(n.min(1 << 22));
    for _ in 0..n {
        v.push(c.u32()?);
    }
    Ok(v)
}

fn read_f64s<const N: usize>(c: &mut Cur<'_>) -> Result<[f64; N]> {
    let mut a = [0.0; N];
    for x in a.iter_mut() {
        *x = c.f64()?;
    }
    Ok(a)
}

fn read_f32s<const N: usize>(c: &mut Cur<'_>) -> Result<[f32; N]> {
    let mut a = [0.0; N];
    for x in a.iter_mut() {
        *x = c.f32()?;
    }
    Ok(a)
}

fn read_f16s<const N: usize>(c: &mut Cur<'_>) -> Result<[f32; N]> {
    let mut a = [0.0; N];
    for x in a.iter_mut() {
        *x = half_to_f32(c.u16()?);
    }
    Ok(a)
}

fn read_i32s<const N: usize>(c: &mut Cur<'_>) -> Result<[i32; N]> {
    let mut a = [0; N];
    for x in a.iter_mut() {
        *x = c.i32()?;
    }
    Ok(a)
}

/// `Sdf_IntegerCompression`: u64 compressed size, then a `TfFastCompression`
/// block holding the integer codec output.
fn read_compressed_ints(c: &mut Cur<'_>, n: usize, width: usize) -> Result<Vec<i64>> {
    let size = c.u64()? as usize;
    let bytes = c.bytes(size)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let max_out = width + (n * 2).div_ceil(8) + n * width;
    let decoded = fast_decompress(bytes, max_out)?;
    decode_ints(&decoded, n, width)
}

fn read_compressed_u32s(c: &mut Cur<'_>, n: usize) -> Result<Vec<u32>> {
    Ok(read_compressed_ints(c, n, 4)?
        .into_iter()
        .map(|x| x as u32)
        .collect())
}

fn read_compressed_i32s(c: &mut Cur<'_>, n: usize) -> Result<Vec<i32>> {
    Ok(read_compressed_ints(c, n, 4)?
        .into_iter()
        .map(|x| x as i32)
        .collect())
}

/// `_ReadTOC`: `u64` count, then `(name[16], start, size)` records.
fn read_toc(data: &[u8], toc_offset: usize) -> Result<Vec<Section>> {
    let mut c = Cur::at(data, toc_offset)?;
    let n = c.u64()? as usize;
    let mut out = Vec::with_capacity(n.min(64));
    for _ in 0..n {
        let name = c.arr::<16>()?;
        let end = name.iter().position(|&b| b == 0).unwrap_or(16);
        let name = String::from_utf8_lossy(&name[..end]).into_owned();
        let start = c.u64()?;
        let size = c.u64()?;
        out.push(Section { name, start, size });
    }
    for s in &out {
        if s.start
            .checked_add(s.size)
            .is_none_or(|e| e > data.len() as u64)
        {
            return malformed(format!("section {} lies past the end of the file", s.name));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_known_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
        assert!((half_to_f32(0x3555) - 0.33325195).abs() < 1e-7);
    }

    #[test]
    fn header_rejects_bad_magic_and_short_files() {
        assert!(matches!(
            CrateFile::parse(vec![0; 10]),
            Err(UsdError::Malformed(_))
        ));
        let mut bytes = vec![0u8; 200];
        bytes[..8].copy_from_slice(b"NOTUSDC!");
        assert!(matches!(
            CrateFile::parse(bytes),
            Err(UsdError::Malformed(_))
        ));
    }

    #[test]
    fn header_rejects_unsupported_version_by_name() {
        let mut bytes = vec![0u8; 200];
        bytes[..8].copy_from_slice(b"PXR-USDC");
        bytes[8] = 0;
        bytes[9] = 3;
        bytes[10] = 0;
        match CrateFile::parse(bytes) {
            Err(UsdError::Unsupported(m)) => assert!(m.contains("0.3.0"), "{m}"),
            other => panic!("expected unsupported version, got {other:?}"),
        }
    }

    #[test]
    fn toc_parses_named_sections() {
        let mut data = vec![0u8; 200];
        let toc = 100usize;
        data[toc..toc + 8].copy_from_slice(&1u64.to_le_bytes());
        data[toc + 8..toc + 12].copy_from_slice(b"PATH");
        data[toc + 24..toc + 32].copy_from_slice(&50u64.to_le_bytes());
        data[toc + 32..toc + 40].copy_from_slice(&8u64.to_le_bytes());
        let secs = read_toc(&data, toc).unwrap();
        assert_eq!(secs.len(), 1);
        assert_eq!(secs[0].name, "PATH");
        assert_eq!(secs[0].start, 50);
        assert_eq!(secs[0].size, 8);
    }

    #[test]
    fn inline_packed_vec3f_and_identity_matrix() {
        let file = CrateFile {
            version: (0, 8, 0),
            data: Vec::new(),
            tokens: Vec::new(),
            strings: Vec::new(),
            fields: Vec::new(),
            fieldsets: Vec::new(),
            paths: Vec::new(),
            specs: Vec::new(),
        };
        // int8 components 1, -2, 3 packed into the low three bytes.
        let low = u32::from_le_bytes([1, (-2i8) as u8, 3, 0]);
        assert_eq!(
            file.inline_packed(T_VEC3F, low).unwrap(),
            Value::Vec3f([1.0, -2.0, 3.0])
        );
        // Diagonal (1, 1, 1, 1) with zero off-diagonal is the 4x4 identity.
        let diag = u32::from_le_bytes([1, 1, 1, 1]);
        match file.inline_packed(T_MATRIX4D, diag).unwrap() {
            Value::Matrix4d(m) => {
                for (k, x) in m.iter().enumerate() {
                    let want = if k % 5 == 0 { 1.0 } else { 0.0 };
                    assert_eq!(*x, want, "entry {k}");
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unknown_type_is_named_not_skipped() {
        let file = CrateFile {
            version: (0, 8, 0),
            data: vec![0u8; 64],
            tokens: Vec::new(),
            strings: Vec::new(),
            fields: Vec::new(),
            fieldsets: Vec::new(),
            paths: Vec::new(),
            specs: Vec::new(),
        };
        // Type id 59 is Spline, which is not decoded.
        let rep = (59u64) << 48 | 8;
        match file.unpack(rep, 0) {
            Err(UsdError::UnsupportedValueType { id, name }) => {
                assert_eq!(id, 59);
                assert_eq!(name, "Spline");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
