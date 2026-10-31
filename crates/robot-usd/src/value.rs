//! The value model shared by the crate reader, the text format and the stage.
//!
//! Half-precision values are widened to `f32` when decoded, so `Value` has no
//! half variants. Matrices and quaternions are stored in OpenUSD's memory
//! order: matrices row-major, quaternions as `(real, i, j, k)`.

/// `SdfSpecifier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specifier {
    /// `def`
    Def,
    /// `over`
    Over,
    /// `class`
    Class,
}

/// `SdfVariability`. The legacy "config" value is read as `Uniform`, as
/// OpenUSD does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variability {
    /// Default: the value may vary over time.
    Varying,
    /// `uniform`
    Uniform,
}

/// A list-editing operation (`SdfListOp`) over items of one kind.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ListOp {
    /// True when the list is explicit (it replaces weaker opinions).
    pub is_explicit: bool,
    /// Items of an explicit list.
    pub explicit: Vec<Value>,
    /// Items added to the list.
    pub added: Vec<Value>,
    /// Items prepended to the list.
    pub prepended: Vec<Value>,
    /// Items appended to the list.
    pub appended: Vec<Value>,
    /// Items removed from the list.
    pub deleted: Vec<Value>,
    /// Items placed in the given order.
    pub ordered: Vec<Value>,
}

impl ListOp {
    /// The items of a single-layer list, in USD's composition order:
    /// explicit items if present, otherwise prepended, added, appended.
    /// Deleted and ordered items are reported as unsupported by the caller.
    pub fn effective_items(&self) -> Vec<Value> {
        if self.is_explicit {
            return self.explicit.clone();
        }
        let mut out = Vec::new();
        out.extend(self.prepended.iter().cloned());
        out.extend(self.added.iter().cloned());
        out.extend(self.appended.iter().cloned());
        out
    }
}

/// `SdfReference` or `SdfPayload`: an external arc.
#[derive(Debug, Clone, PartialEq)]
pub struct Arc {
    /// The asset path (layer identifier), possibly empty for an internal arc.
    pub asset_path: String,
    /// The target prim path, possibly empty for the default prim.
    pub prim_path: String,
    /// Layer offset (`offset`, `scale`); `(0, 1)` when absent.
    pub layer_offset: (f64, f64),
    /// Custom data of a reference (empty for payloads).
    pub custom_data: Vec<(String, Value)>,
}

/// A typed USD value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `bool`
    Bool(bool),
    /// `uchar`
    UChar(u8),
    /// `int`
    Int(i32),
    /// `uint`
    UInt(u32),
    /// `int64`
    Int64(i64),
    /// `uint64`
    UInt64(u64),
    /// `float` (also the widened form of `half`)
    Float(f32),
    /// `double`
    Double(f64),
    /// `string`
    String(String),
    /// `token`
    Token(String),
    /// `asset`
    AssetPath(String),
    /// `path`
    Path(String),
    /// `float2`
    Vec2f([f32; 2]),
    /// `float3`
    Vec3f([f32; 3]),
    /// `float4`
    Vec4f([f32; 4]),
    /// `double2`
    Vec2d([f64; 2]),
    /// `double3`
    Vec3d([f64; 3]),
    /// `double4`
    Vec4d([f64; 4]),
    /// `int2`
    Vec2i([i32; 2]),
    /// `int3`
    Vec3i([i32; 3]),
    /// `int4`
    Vec4i([i32; 4]),
    /// `quatf`: `(real, i, j, k)`
    Quatf([f32; 4]),
    /// `quatd`: `(real, i, j, k)`
    Quatd([f64; 4]),
    /// `matrix2d`, row-major
    Matrix2d(Box<[f64; 4]>),
    /// `matrix3d`, row-major
    Matrix3d(Box<[f64; 9]>),
    /// `matrix4d`, row-major
    Matrix4d(Box<[f64; 16]>),
    /// A homogeneous array (`VtArray<T>`) or a `*Vector`; elements share a kind.
    Array(Vec<Value>),
    /// A list operation over tokens, strings, paths, ints, references or payloads.
    ListOp(Box<ListOp>),
    /// A `VtDictionary`, in stored order.
    Dictionary(Vec<(String, Value)>),
    /// Time samples: `(time, value)` pairs in increasing time.
    TimeSamples(Vec<(f64, Value)>),
    /// A reference arc.
    Reference(Box<Arc>),
    /// A payload arc.
    Payload(Box<Arc>),
    /// `SdfSpecifier`
    Specifier(Specifier),
    /// `SdfVariability`
    Variability(Variability),
    /// An `SdfValueBlock`: an authored "no value".
    ValueBlock,
    /// An `SdfLayerOffset` as `(offset, scale)`.
    LayerOffset(f64, f64),
}

impl Value {
    /// Returns the string of a `Token`, `String`, `AssetPath` or `Path`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Token(s) | Value::String(s) | Value::AssetPath(s) | Value::Path(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the number of a `Double` or `Float`, widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Double(d) => Some(*d),
            Value::Float(f) => Some(*f as f64),
            _ => None,
        }
    }

    /// Returns the elements of an `Array`.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the tokens or strings of an `Array` as owned strings, or `None`
    /// if an element is not a string kind.
    pub fn string_items(&self) -> Option<Vec<String>> {
        self.as_array()?
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect()
    }
}
