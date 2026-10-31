//! USDA text: a writer for [`Stage`] and a reader for the subset the writer
//! emits, plus the standard constructs the G1 stage needs.
//!
//! Syntax written (standard USDA where one exists):
//!
//! ```text
//! #usda 1.0
//! (
//!     defaultPrim = "g1"            // metadata: typed with a leading keyword
//! )
//! def Xform "g1" (
//!     kind = "component"
//!     apiSchemas = ["PhysicsArticulationRootAPI"]
//! ) {
//!     uniform token[] xformOpOrder = ["xformOp:translate"]
//!     double3 xformOp:translate = (0, 0, 0)
//!     rel physics:body0 = </g1/pelvis>
//!     float attr.timeSamples = { 0: 1.0, 1: 2.0 }
//! }
//! ```
//!
//! Prim and layer metadata values carry an explicit type keyword (`token`,
//! `string`, `double`, `dictionary`, ...), because the reader has no schema to
//! type an untyped `key = value` pair. Dictionary entries are `type key = value`
//! separated by whitespace. A list operation is written and read back as an
//! explicit list, so composed stages round-trip exactly. Layer offsets, arcs
//! and variant sets are not written: a composed [`Stage`] has none.

use crate::error::{Result, UsdError};
use crate::layer::{Layer, Spec, SpecKind};
use crate::stage::{Prim, Property, PropertyKind, Stage};
use crate::value::{ListOp, Specifier, Value, Variability};
use std::fmt::Write as _;

// ---------------------------------------------------------------- writer

/// Writes a stage as USDA text. The stage is written flattened: its layer
/// stack is not recorded, because composition has already merged it.
pub fn write_usda(stage: &Stage) -> String {
    let mut out = String::from("#usda 1.0\n");
    if !stage.metadata.is_empty() {
        out.push_str("(\n");
        for (k, v) in &stage.metadata {
            let _ = writeln!(out, "    {}", metadata_entry(k, v));
        }
        out.push_str(")\n");
    }
    out.push('\n');
    for p in &stage.prims {
        write_prim(&mut out, p, 0);
    }
    out
}

fn indent(n: usize) -> String {
    "    ".repeat(n)
}

fn write_prim(out: &mut String, p: &Prim, depth: usize) {
    let pad = indent(depth);
    let spec = match p.specifier.unwrap_or(Specifier::Over) {
        Specifier::Def => "def",
        Specifier::Over => "over",
        Specifier::Class => "class",
    };
    let ty = p
        .type_name
        .as_deref()
        .map(|t| format!("{t} "))
        .unwrap_or_default();
    let _ = write!(out, "{pad}{spec} {ty}\"{}\"", escape(&p.name));
    let mut meta: Vec<String> = Vec::new();
    if let Some(k) = &p.kind {
        meta.push(format!("token kind = \"{}\"", escape(k)));
    }
    if !p.api_schemas.is_empty() {
        let items: Vec<String> = p
            .api_schemas
            .iter()
            .map(|s| format!("\"{}\"", escape(s)))
            .collect();
        meta.push(format!("apiSchemas = [{}]", items.join(", ")));
    }
    for (k, v) in &p.metadata {
        meta.push(metadata_entry(k, v));
    }
    if !meta.is_empty() {
        out.push_str(" (\n");
        for m in &meta {
            let _ = writeln!(out, "{pad}    {m}");
        }
        let _ = write!(out, "{pad})");
    }
    out.push_str(" {\n");
    for prop in &p.properties {
        write_property(out, prop, depth + 1);
    }
    for c in &p.children {
        write_prim(out, c, depth + 1);
    }
    let _ = writeln!(out, "{pad}}}");
}

fn write_property(out: &mut String, prop: &Property, depth: usize) {
    let pad = indent(depth);
    let mut line = String::new();
    match prop.kind {
        PropertyKind::Relationship => {
            let var = match prop.variability {
                Some(Variability::Uniform) => "uniform ",
                Some(Variability::Varying) => "varying ",
                None => "",
            };
            let _ = write!(line, "{pad}{var}rel {}", prop.name);
            if !prop.targets.is_empty() {
                let _ = write!(line, " = {}", connection_text(&prop.targets));
            }
        }
        PropertyKind::Attribute => {
            let (ty, text) = match (&prop.type_name, &prop.default) {
                (Some(t), Some(v)) => (t.clone(), value_text(v)),
                (Some(t), None) => (t.clone(), None),
                (None, Some(v)) => (type_name_of(v), value_text(v)),
                (None, None) => ("token".to_owned(), None),
            };
            let var = match prop.variability {
                Some(Variability::Uniform) => "uniform ",
                Some(Variability::Varying) => "varying ",
                None => "",
            };
            let _ = write!(line, "{pad}{var}{ty} {}", prop.name);
            if !prop.connections.is_empty() {
                let _ = write!(line, ".connect = {}", connection_text(&prop.connections));
            } else if let Some(samples) = &prop.time_samples {
                let items: Vec<String> = samples
                    .iter()
                    .map(|(t, v)| {
                        format!(
                            "{}: {}",
                            fmt_f64(*t),
                            value_text(v).unwrap_or_else(none_text)
                        )
                    })
                    .collect();
                let _ = write!(line, ".timeSamples = {{ {} }}", items.join(", "));
            } else {
                let _ = write!(line, " = {}", text.unwrap_or_else(none_text));
            }
        }
    }
    if !prop.metadata.is_empty() {
        let entries: Vec<String> = prop
            .metadata
            .iter()
            .map(|(k, v)| metadata_entry(k, v))
            .collect();
        let _ = write!(
            line,
            " (\n{pad}    {}\n{pad})",
            entries.join(&format!("\n{pad}    "))
        );
    }
    out.push_str(&line);
    out.push('\n');
}

fn none_text() -> String {
    "None".to_owned()
}

/// A metadata entry `keyword key = value`.
fn metadata_entry(key: &str, v: &Value) -> String {
    let kw = type_name_of(v);
    format!("{kw} {key} = {}", value_text(v).unwrap_or_else(none_text))
}

fn value_text(v: &Value) -> Option<String> {
    Some(match v {
        Value::Bool(b) => b.to_string(),
        Value::UChar(x) => x.to_string(),
        Value::Int(x) => x.to_string(),
        Value::UInt(x) => x.to_string(),
        Value::Int64(x) => x.to_string(),
        Value::UInt64(x) => x.to_string(),
        Value::Float(x) => fmt_f32(*x),
        Value::Double(x) => fmt_f64(*x),
        Value::String(s) | Value::Token(s) => format!("\"{}\"", escape(s)),
        Value::AssetPath(s) => format!("@{s}@"),
        Value::Path(s) => format!("<{s}>"),
        Value::Vec2f(a) => tuple(a.iter().map(|x| fmt_f32(*x))),
        Value::Vec3f(a) => tuple(a.iter().map(|x| fmt_f32(*x))),
        Value::Vec4f(a) => tuple(a.iter().map(|x| fmt_f32(*x))),
        Value::Vec2d(a) => tuple(a.iter().map(|x| fmt_f64(*x))),
        Value::Vec3d(a) => tuple(a.iter().map(|x| fmt_f64(*x))),
        Value::Vec4d(a) => tuple(a.iter().map(|x| fmt_f64(*x))),
        Value::Vec2i(a) => tuple(a.iter().map(|x| x.to_string())),
        Value::Vec3i(a) => tuple(a.iter().map(|x| x.to_string())),
        Value::Vec4i(a) => tuple(a.iter().map(|x| x.to_string())),
        Value::Quatf(a) => tuple(a.iter().map(|x| fmt_f32(*x))),
        Value::Quatd(a) => tuple(a.iter().map(|x| fmt_f64(*x))),
        Value::Matrix2d(m) => matrix(&m[..], 2),
        Value::Matrix3d(m) => matrix(&m[..], 3),
        Value::Matrix4d(m) => matrix(&m[..], 4),
        Value::Array(items) => {
            let parts: Option<Vec<String>> = items.iter().map(value_text).collect();
            format!("[{}]", parts?.join(", "))
        }
        Value::ListOp(op) => {
            let parts: Option<Vec<String>> = op.effective_items().iter().map(value_text).collect();
            format!("[{}]", parts?.join(", "))
        }
        Value::Dictionary(entries) => {
            let mut s = String::from("{");
            for (k, v) in entries {
                let _ = write!(s, " {} {k} = {}", type_name_of(v), value_text(v)?);
            }
            s.push_str(" }");
            s
        }
        Value::TimeSamples(samples) => {
            let items: Option<Vec<String>> = samples
                .iter()
                .map(|(t, v)| Some(format!("{}: {}", fmt_f64(*t), value_text(v)?)))
                .collect();
            format!("{{ {} }}", items?.join(", "))
        }
        Value::ValueBlock => return None,
        Value::Reference(_) | Value::Payload(_) | Value::LayerOffset(..) => return None,
        Value::Specifier(_) | Value::Variability(_) => return None,
    })
}

fn connection_text(targets: &[String]) -> String {
    match targets.len() {
        1 => format!("<{}>", targets[0]),
        _ => format!(
            "[{}]",
            targets
                .iter()
                .map(|t| format!("<{t}>"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn matrix(m: &[f64], n: usize) -> String {
    let rows: Vec<String> = (0..n)
        .map(|r| tuple(m[r * n..r * n + n].iter().map(|x| fmt_f64(*x))))
        .collect();
    format!("({})", rows.join(", "))
}

fn tuple(items: impl Iterator<Item = String>) -> String {
    format!("({})", items.collect::<Vec<_>>().join(", "))
}

/// Shortest text that reads back to the same `f32`, always with a decimal
/// point or exponent so the reader types it as a float.
fn fmt_f32(x: f32) -> String {
    if x.is_finite() {
        format!("{x:?}")
    } else {
        format!("{x}")
    }
}

fn fmt_f64(x: f64) -> String {
    if x.is_finite() {
        format!("{x:?}")
    } else {
        format!("{x}")
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The USD type name for a value, used when no `typeName` is authored.
fn type_name_of(v: &Value) -> String {
    match v {
        Value::Bool(_) => "bool",
        Value::UChar(_) => "uchar",
        Value::Int(_) => "int",
        Value::UInt(_) => "uint",
        Value::Int64(_) => "int64",
        Value::UInt64(_) => "uint64",
        Value::Float(_) => "float",
        Value::Double(_) => "double",
        Value::String(_) => "string",
        Value::Token(_) => "token",
        Value::AssetPath(_) => "asset",
        Value::Path(_) => "path",
        Value::Vec2f(_) => "float2",
        Value::Vec3f(_) => "float3",
        Value::Vec4f(_) => "float4",
        Value::Vec2d(_) => "double2",
        Value::Vec3d(_) => "double3",
        Value::Vec4d(_) => "double4",
        Value::Vec2i(_) => "int2",
        Value::Vec3i(_) => "int3",
        Value::Vec4i(_) => "int4",
        Value::Quatf(_) => "quatf",
        Value::Quatd(_) => "quatd",
        Value::Matrix2d(_) => "matrix2d",
        Value::Matrix3d(_) => "matrix3d",
        Value::Matrix4d(_) => "matrix4d",
        Value::Array(items) => {
            return match items.first() {
                Some(first) => format!("{}[]", type_name_of(first)),
                None => "token[]".to_owned(),
            }
        }
        Value::ListOp(_) => "token[]",
        Value::Dictionary(_) => "dictionary",
        Value::TimeSamples(_) => "double",
        Value::Reference(_) | Value::Payload(_) | Value::LayerOffset(..) => "token",
        Value::Specifier(_) => "token",
        Value::Variability(_) => "token",
        Value::ValueBlock => "token",
    }
    .to_owned()
}

// ---------------------------------------------------------------- reader

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(String),
    Str(String),
    Asset(String),
    Path(String),
    Punct(char),
}

#[derive(Debug, Clone)]
struct Lexed {
    tok: Tok,
    line: usize,
}

fn lex(text: &str) -> Result<Vec<Lexed>> {
    let mut out = Vec::new();
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut line = 1usize;
    let err = |line: usize, m: &str| UsdError::Parse {
        line,
        message: m.to_owned(),
    };
    while i < cs.len() {
        let c = cs[i];
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c.is_whitespace() {
            i += 1;
        } else if c == '#' || (c == '/' && cs.get(i + 1) == Some(&'/')) {
            while i < cs.len() && cs[i] != '\n' {
                i += 1;
            }
        } else if c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                match cs.get(i) {
                    None | Some('\n') => return Err(err(line, "unterminated string")),
                    Some('"') => {
                        i += 1;
                        break;
                    }
                    Some('\\') => {
                        s.push(*cs.get(i + 1).ok_or_else(|| err(line, "bad escape"))?);
                        i += 2;
                    }
                    Some(&ch) => {
                        s.push(ch);
                        i += 1;
                    }
                }
            }
            out.push(Lexed {
                tok: Tok::Str(s),
                line,
            });
        } else if c == '@' {
            let start = i + 1;
            i += 1;
            while i < cs.len() && cs[i] != '@' {
                i += 1;
            }
            if i >= cs.len() {
                return Err(err(line, "unterminated asset path"));
            }
            out.push(Lexed {
                tok: Tok::Asset(cs[start..i].iter().collect()),
                line,
            });
            i += 1;
        } else if c == '<' {
            let start = i + 1;
            i += 1;
            while i < cs.len() && cs[i] != '>' {
                i += 1;
            }
            if i >= cs.len() {
                return Err(err(line, "unterminated path"));
            }
            out.push(Lexed {
                tok: Tok::Path(cs[start..i].iter().collect()),
                line,
            });
            i += 1;
        } else if c.is_ascii_digit()
            || ((c == '-' || c == '+' || c == '.')
                && cs
                    .get(i + 1)
                    .is_some_and(|d| d.is_ascii_digit() || *d == '.'))
        {
            let start = i;
            i += 1;
            while i < cs.len() {
                let d = cs[i];
                let exp_sign = (d == '-' || d == '+') && matches!(cs[i - 1], 'e' | 'E');
                if d.is_ascii_alphanumeric() || d == '.' || exp_sign {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Lexed {
                tok: Tok::Num(cs[start..i].iter().collect()),
                line,
            });
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < cs.len()
                && (cs[i].is_ascii_alphanumeric() || matches!(cs[i], '_' | ':' | '.'))
            {
                i += 1;
            }
            out.push(Lexed {
                tok: Tok::Ident(cs[start..i].iter().collect()),
                line,
            });
        } else if "(){}[],=:".contains(c) {
            out.push(Lexed {
                tok: Tok::Punct(c),
                line,
            });
            i += 1;
        } else {
            return Err(err(line, &format!("unexpected character {c:?}")));
        }
    }
    Ok(out)
}

/// A parsed value before its declared type is applied.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    Num(String),
    Bool(bool),
    Str(String),
    Asset(String),
    Path(String),
    Tuple(Vec<Node>),
    Array(Vec<Node>),
    Dict(Vec<(String, String, Node)>),
    Samples(Vec<(String, Node)>),
    None,
}

struct Parser {
    toks: Vec<Lexed>,
    pos: usize,
    id: String,
}

impl Parser {
    fn err<T>(&self, m: impl Into<String>) -> Result<T> {
        let line = self
            .toks
            .get(self.pos)
            .or(self.toks.last())
            .map(|t| t.line)
            .unwrap_or(1);
        Err(UsdError::Parse {
            line,
            message: m.into(),
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn next(&mut self) -> Result<Tok> {
        let t = self.toks.get(self.pos).map(|t| t.tok.clone());
        self.pos += 1;
        match t {
            Some(t) => Ok(t),
            None => self.err("unexpected end of file"),
        }
    }

    fn expect(&mut self, c: char) -> Result<()> {
        match self.next()? {
            Tok::Punct(p) if p == c => Ok(()),
            other => self.err(format!("expected '{c}', found {other:?}")),
        }
    }

    fn is_punct(&self, c: char) -> bool {
        matches!(self.peek(), Some(Tok::Punct(p)) if *p == c)
    }

    /// A type keyword with an optional `[]` array suffix, for example `float3[]`.
    fn type_word(&mut self) -> Result<String> {
        let mut t = self.ident()?;
        if self.is_punct('[') {
            self.pos += 1;
            self.expect(']')?;
            t.push_str("[]");
        }
        Ok(t)
    }

    fn ident(&mut self) -> Result<String> {
        match self.next()? {
            Tok::Ident(s) => Ok(s),
            other => self.err(format!("expected a name, found {other:?}")),
        }
    }

    /// A value: numbers, strings, tuples, arrays, dictionaries and samples.
    fn value(&mut self) -> Result<Node> {
        match self.next()? {
            Tok::Num(n) => Ok(Node::Num(n)),
            Tok::Str(s) => Ok(Node::Str(s)),
            Tok::Asset(s) => Ok(Node::Asset(s)),
            Tok::Path(s) => Ok(Node::Path(s)),
            Tok::Ident(s) if s == "true" => Ok(Node::Bool(true)),
            Tok::Ident(s) if s == "false" => Ok(Node::Bool(false)),
            Tok::Ident(s) if s == "None" => Ok(Node::None),
            Tok::Ident(s) if s == "inf" || s == "nan" => Ok(Node::Num(s)),
            Tok::Punct('(') => {
                let mut items = Vec::new();
                while !self.is_punct(')') {
                    items.push(self.value()?);
                    if self.is_punct(',') {
                        self.pos += 1;
                    }
                }
                self.pos += 1;
                Ok(Node::Tuple(items))
            }
            Tok::Punct('[') => {
                let mut items = Vec::new();
                while !self.is_punct(']') {
                    items.push(self.value()?);
                    if self.is_punct(',') {
                        self.pos += 1;
                    }
                }
                self.pos += 1;
                Ok(Node::Array(items))
            }
            Tok::Punct('{') => {
                // Time samples start with a number and a colon; dictionaries with a type name.
                if matches!(self.peek(), Some(Tok::Num(_))) {
                    let mut samples = Vec::new();
                    while !self.is_punct('}') {
                        let t = match self.next()? {
                            Tok::Num(n) => n,
                            other => return self.err(format!("expected a time, found {other:?}")),
                        };
                        self.expect(':')?;
                        samples.push((t, self.value()?));
                        if self.is_punct(',') {
                            self.pos += 1;
                        }
                    }
                    self.pos += 1;
                    return Ok(Node::Samples(samples));
                }
                let mut entries = Vec::new();
                while !self.is_punct('}') {
                    let ty = self.type_word()?;
                    let key = self.ident()?;
                    self.expect('=')?;
                    entries.push((ty, key, self.value()?));
                    if self.is_punct(',') {
                        self.pos += 1;
                    }
                }
                self.pos += 1;
                Ok(Node::Dict(entries))
            }
            other => self.err(format!("expected a value, found {other:?}")),
        }
    }

    /// `( key = value ... )` metadata, each entry `[type] key = value`.
    /// Returns `(key, type keyword, value)`; the keyword is the key itself when absent.
    fn metadata(&mut self) -> Result<Vec<(String, String, Node)>> {
        let mut out = Vec::new();
        self.expect('(')?;
        while !self.is_punct(')') {
            let first = self.type_word()?;
            let (kw, key) = if matches!(self.peek(), Some(Tok::Ident(_))) {
                (first, self.ident()?)
            } else {
                (first.clone(), first)
            };
            self.expect('=')?;
            let node = self.value()?;
            out.push((key, kw, node));
        }
        self.pos += 1;
        Ok(out)
    }

    fn file(&mut self) -> Result<Layer> {
        let mut specs = Vec::new();
        let mut root_children = Vec::new();
        let mut root_meta = Vec::new();
        if self.is_punct('(') {
            root_meta = self.metadata()?;
        }
        while self.peek().is_some() {
            let (prim, _) = self.prim("")?;
            root_children.push(prim.name.clone());
            specs.extend(prim.specs);
        }
        let mut root_fields = root_meta
            .into_iter()
            .map(|(k, kw, n)| Ok((k.clone(), meta_value(&kw, &k, &n)?)))
            .collect::<Result<Vec<(String, Value)>>>()?;
        root_fields.push(("primChildren".into(), names_array(&root_children)));
        specs.insert(
            0,
            Spec {
                path: "/".into(),
                kind: SpecKind::PseudoRoot,
                fields: root_fields,
            },
        );
        Ok(Layer {
            identifier: self.id.clone(),
            specs,
        })
    }

    /// Parses one prim; returns its specs and the prim record.
    fn prim(&mut self, parent: &str) -> Result<(ParsedPrim, ())> {
        let spec_kw = self.ident()?;
        let specifier = match spec_kw.as_str() {
            "def" => Specifier::Def,
            "over" => Specifier::Over,
            "class" => Specifier::Class,
            other => return self.err(format!("expected def, over or class, found {other}")),
        };
        let mut type_name = None;
        let name = match self.next()? {
            Tok::Str(s) => s,
            Tok::Ident(t) => {
                type_name = Some(t);
                match self.next()? {
                    Tok::Str(s) => s,
                    other => return self.err(format!("expected a prim name, found {other:?}")),
                }
            }
            other => return self.err(format!("expected a prim name, found {other:?}")),
        };
        let path = if parent.is_empty() {
            format!("/{name}")
        } else {
            format!("{parent}/{name}")
        };
        let mut fields = vec![("specifier".to_owned(), Value::Specifier(specifier))];
        if let Some(t) = &type_name {
            fields.push(("typeName".into(), Value::Token(t.clone())));
        }
        let mut prim_specs = Vec::new();
        if self.is_punct('(') {
            for (k, kw, n) in self.metadata()? {
                if k == "apiSchemas" {
                    let items = meta_value(&kw, &k, &n)?;
                    let op = ListOp {
                        is_explicit: true,
                        explicit: items.as_array().map(<[Value]>::to_vec).unwrap_or_default(),
                        ..ListOp::default()
                    };
                    fields.push((k, Value::ListOp(Box::new(op))));
                } else {
                    fields.push((k.clone(), meta_value(&kw, &k, &n)?));
                }
            }
        }
        self.expect('{')?;
        let mut children = Vec::new();
        let mut properties = Vec::new();
        let mut prop_specs = Vec::new();
        while !self.is_punct('}') {
            if matches!(self.peek(), Some(Tok::Ident(s)) if matches!(s.as_str(), "def" | "over" | "class"))
            {
                let (child, _) = self.prim(&path)?;
                children.push(child.name.clone());
                prim_specs.extend(child.specs);
            } else {
                let p = self.property(&path)?;
                properties.push(p.name.clone());
                prop_specs.push(p.spec);
            }
        }
        self.pos += 1;
        fields.push(("primChildren".into(), names_array(&children)));
        fields.push(("properties".into(), names_array(&properties)));
        let mut specs = vec![Spec {
            path: path.clone(),
            kind: SpecKind::Prim,
            fields,
        }];
        specs.extend(prop_specs);
        specs.extend(prim_specs);
        Ok((ParsedPrim { name, specs }, ()))
    }

    /// The target of a relationship: one path, or a list of paths.
    fn rel_targets(&mut self) -> Result<Vec<String>> {
        match self.value()? {
            Node::Path(p) => Ok(vec![p]),
            Node::Array(items) => items
                .into_iter()
                .map(|n| match n {
                    Node::Path(p) => Ok(p),
                    other => self.err(format!("relationship target {other:?} is not a path")),
                })
                .collect(),
            other => self.err(format!("relationship value {other:?} is not a path")),
        }
    }

    fn property(&mut self, owner: &str) -> Result<ParsedProp> {
        let mut variability = None;
        let mut first = self.type_word()?;
        if matches!(first.as_str(), "uniform" | "varying" | "custom") {
            variability = match first.as_str() {
                "uniform" => Some(Variability::Uniform),
                "varying" => Some(Variability::Varying),
                _ => None,
            };
            first = self.type_word()?;
        }
        if first == "rel" {
            let name = self.ident()?;
            let mut meta = Vec::new();
            let targets: Vec<String> = if !self.is_punct('=') {
                Vec::new()
            } else {
                self.pos += 1;
                self.rel_targets()?
            };
            if self.is_punct('(') {
                meta = self.metadata()?;
            }
            let op = ListOp {
                is_explicit: true,
                explicit: targets.into_iter().map(Value::Path).collect(),
                ..ListOp::default()
            };
            let path = format!("{owner}.{name}");
            let mut fields = vec![("targetPaths".to_owned(), Value::ListOp(Box::new(op)))];
            if let Some(v) = variability {
                fields.push(("variability".to_owned(), Value::Variability(v)));
            }
            for (k, kw, n) in meta {
                fields.push((k.clone(), meta_value(&kw, &k, &n)?));
            }
            return Ok(ParsedProp {
                name,
                spec: Spec {
                    path,
                    kind: SpecKind::Relationship,
                    fields,
                },
            });
        }
        // Attribute: `type name [.timeSamples] = value`, with an optional `uniform`.
        let type_name = first;
        let name_tok = self.ident()?;
        let (name, samples, connect) = if let Some(n) = name_tok.strip_suffix(".timeSamples") {
            (n.to_owned(), true, false)
        } else if let Some(n) = name_tok.strip_suffix(".connect") {
            (n.to_owned(), false, true)
        } else {
            (name_tok, false, false)
        };
        self.expect('=')?;
        let node = self.value()?;
        let path = format!("{owner}.{name}");
        let mut fields = vec![("typeName".to_owned(), Value::Token(type_name.clone()))];
        if let Some(v) = variability {
            fields.push(("variability".into(), Value::Variability(v)));
        }
        if connect {
            let targets: Vec<Value> = match node {
                Node::Path(p) => vec![Value::Path(p)],
                Node::Array(items) => items
                    .into_iter()
                    .map(|n| match n {
                        Node::Path(p) => Ok(Value::Path(p)),
                        other => self.err(format!("connection {other:?} is not a path")),
                    })
                    .collect::<Result<_>>()?,
                other => return self.err(format!("connection value {other:?} is not a path")),
            };
            let op = ListOp {
                is_explicit: true,
                explicit: targets,
                ..ListOp::default()
            };
            fields.push(("connectionPaths".into(), Value::ListOp(Box::new(op))));
            if self.is_punct('(') {
                for (k, kw, n) in self.metadata()? {
                    fields.push((k.clone(), meta_value(&kw, &k, &n)?));
                }
            }
            return Ok(ParsedProp {
                name,
                spec: Spec {
                    path,
                    kind: SpecKind::Attribute,
                    fields,
                },
            });
        }
        if samples {
            let Node::Samples(list) = node else {
                return self.err("timeSamples value must be a { time: value } list");
            };
            let mut out = Vec::new();
            for (t, n) in list {
                let time: f64 = t.parse().map_err(|_| UsdError::Parse {
                    line: 0,
                    message: format!("bad time {t}"),
                })?;
                out.push((time, node_to_typed(&n, &type_name)?));
            }
            fields.push(("timeSamples".into(), Value::TimeSamples(out)));
        } else if !matches!(node, Node::None) {
            fields.push(("default".into(), node_to_typed(&node, &type_name)?));
        }
        if self.is_punct('(') {
            for (k, kw, n) in self.metadata()? {
                fields.push((k.clone(), meta_value(&kw, &k, &n)?));
            }
        }
        Ok(ParsedProp {
            name,
            spec: Spec {
                path,
                kind: SpecKind::Attribute,
                fields,
            },
        })
    }
}

struct ParsedPrim {
    name: String,
    specs: Vec<Spec>,
}

struct ParsedProp {
    name: String,
    spec: Spec,
}

fn names_array(names: &[String]) -> Value {
    Value::Array(names.iter().map(|n| Value::Token(n.clone())).collect())
}

/// Parses a USDA document into one layer.
pub fn parse_layer(text: &str, id: &str) -> Result<Layer> {
    let toks = lex(text)?;
    let mut p = Parser {
        toks,
        pos: 0,
        id: id.to_owned(),
    };
    if !text.trim_start().starts_with("#usda") {
        return Err(UsdError::Parse {
            line: 1,
            message: "missing #usda header".into(),
        });
    }
    let mut layer = p.file()?;
    // The writer emits prims with `over` for unspecified specifiers; the reader
    // keeps what the file says.
    layer.identifier = id.to_owned();
    Ok(layer)
}

/// Parses USDA text and composes it as a single-layer stage.
pub fn parse_stage(text: &str) -> Result<Stage> {
    let layer = parse_layer(text, "<usda>")?;
    Stage::from_layers(vec![layer])
}

/// Applies a declared USD type name to a parsed value.
fn node_to_typed(n: &Node, ty: &str) -> Result<Value> {
    if let Some(elem) = ty.strip_suffix("[]") {
        return match n {
            Node::Array(items) => Ok(Value::Array(
                items
                    .iter()
                    .map(|i| node_to_typed(i, elem))
                    .collect::<Result<_>>()?,
            )),
            other => Err(UsdError::Parse {
                line: 0,
                message: format!("{ty} expects an array, found {other:?}"),
            }),
        };
    }
    if let Some(kind) = matrix_kind(ty) {
        return matrix_from(n, kind);
    }
    if let Some((dims, base)) = vector_kind(ty) {
        return vector_from(n, dims, base);
    }
    Ok(match ty {
        "bool" => {
            Value::Bool(matches!(n, Node::Bool(true)) || matches!(n, Node::Num(x) if x != "0"))
        }
        "uchar" => Value::UChar(num(n)? as u8),
        "int" => Value::Int(num(n)? as i32),
        "uint" => Value::UInt(num(n)? as u32),
        "int64" => Value::Int64(num(n)? as i64),
        "uint64" => Value::UInt64(num(n)? as u64),
        "float" | "half" => Value::Float(num(n)? as f32),
        "double" => Value::Double(num(n)?),
        "string" => Value::String(text(n)?),
        "token" => Value::Token(text(n)?),
        "asset" => match n {
            Node::Asset(a) => Value::AssetPath(a.clone()),
            other => {
                return Err(UsdError::Parse {
                    line: 0,
                    message: format!("asset expects @path@, found {other:?}"),
                })
            }
        },
        "path" => match n {
            Node::Path(p) => Value::Path(p.clone()),
            other => {
                return Err(UsdError::Parse {
                    line: 0,
                    message: format!("path expects <path>, found {other:?}"),
                })
            }
        },
        "quatf" | "quatd" => {
            let items = match n {
                Node::Tuple(items) if items.len() == 4 => items,
                other => {
                    return Err(UsdError::Parse {
                        line: 0,
                        message: format!("{ty} expects 4 components, found {other:?}"),
                    })
                }
            };
            let xs: Vec<f64> = items.iter().map(num).collect::<Result<_>>()?;
            if ty == "quatf" {
                Value::Quatf([xs[0] as f32, xs[1] as f32, xs[2] as f32, xs[3] as f32])
            } else {
                Value::Quatd([xs[0], xs[1], xs[2], xs[3]])
            }
        }
        "dictionary" => node_to_value(n)?,
        other => return Err(UsdError::Unsupported(format!("type {other} in USDA text"))),
    })
}

/// A metadata value: typed by its keyword, or untyped when the key is its own keyword.
fn meta_value(kw: &str, key: &str, n: &Node) -> Result<Value> {
    if kw == key {
        node_to_value(n)
    } else {
        node_to_typed(n, kw)
    }
}

/// Reads an untyped value from its syntax alone.
fn node_to_value(n: &Node) -> Result<Value> {
    Ok(match n {
        Node::Bool(b) => Value::Bool(*b),
        Node::Str(s) => Value::String(s.clone()),
        Node::Asset(a) => Value::AssetPath(a.clone()),
        Node::Path(p) => Value::Path(p.clone()),
        Node::Num(x) => Value::Double(x.parse().map_err(|_| UsdError::Parse {
            line: 0,
            message: format!("bad number {x}"),
        })?),
        Node::Array(items) => Value::Array(items.iter().map(node_to_value).collect::<Result<_>>()?),
        Node::Dict(entries) => Value::Dictionary(
            entries
                .iter()
                .map(|(ty, k, v)| Ok((k.clone(), node_to_typed(v, ty)?)))
                .collect::<Result<_>>()?,
        ),
        Node::Tuple(items) => Value::Array(items.iter().map(node_to_value).collect::<Result<_>>()?),
        Node::Samples(_) | Node::None => Value::ValueBlock,
    })
}

fn num(n: &Node) -> Result<f64> {
    match n {
        Node::Num(x) => x.parse::<f64>().map_err(|_| UsdError::Parse {
            line: 0,
            message: format!("bad number {x}"),
        }),
        Node::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(UsdError::Parse {
            line: 0,
            message: format!("expected a number, found {other:?}"),
        }),
    }
}

fn text(n: &Node) -> Result<String> {
    match n {
        Node::Str(s) => Ok(s.clone()),
        other => Err(UsdError::Parse {
            line: 0,
            message: format!("expected a string, found {other:?}"),
        }),
    }
}

/// `float3`, `double2`, `int4` (base then dims) and `point3f`, `color3f`,
/// `texCoord2f` (dims then base) to (dims, base). Quaternions and matrices are
/// excluded.
fn vector_kind(ty: &str) -> Option<(usize, char)> {
    if ty.starts_with("quat") || ty.starts_with("matrix") {
        return None;
    }
    let last = ty.chars().last()?;
    if last.is_ascii_digit() {
        let dims = last.to_digit(10)? as usize;
        let base = match &ty[..ty.len() - 1] {
            "float" | "half" => 'f',
            "double" => 'd',
            "int" => 'i',
            _ => return None,
        };
        return (2..=4).contains(&dims).then_some((dims, base));
    }
    let bytes = ty.as_bytes();
    if bytes.len() < 2 || !matches!(last, 'f' | 'd' | 'i' | 'h') {
        return None;
    }
    let dims = (bytes[bytes.len() - 2] as char).to_digit(10)? as usize;
    let base = if last == 'h' { 'f' } else { last };
    (2..=4).contains(&dims).then_some((dims, base))
}

fn matrix_kind(ty: &str) -> Option<usize> {
    match ty {
        "matrix2d" => Some(2),
        "matrix3d" => Some(3),
        "matrix4d" => Some(4),
        _ => None,
    }
}

fn vector_from(n: &Node, dims: usize, base: char) -> Result<Value> {
    let items = match n {
        Node::Tuple(items) if items.len() == dims => items,
        other => {
            return Err(UsdError::Parse {
                line: 0,
                message: format!("{dims}-component vector expected, found {other:?}"),
            })
        }
    };
    let xs: Vec<f64> = items.iter().map(num).collect::<Result<_>>()?;
    Ok(match (dims, base) {
        (2, 'f') => Value::Vec2f([xs[0] as f32, xs[1] as f32]),
        (3, 'f') => Value::Vec3f([xs[0] as f32, xs[1] as f32, xs[2] as f32]),
        (4, 'f') => Value::Vec4f([xs[0] as f32, xs[1] as f32, xs[2] as f32, xs[3] as f32]),
        (2, 'd') => Value::Vec2d([xs[0], xs[1]]),
        (3, 'd') => Value::Vec3d([xs[0], xs[1], xs[2]]),
        (4, 'd') => Value::Vec4d([xs[0], xs[1], xs[2], xs[3]]),
        (2, _) => Value::Vec2i([xs[0] as i32, xs[1] as i32]),
        (3, _) => Value::Vec3i([xs[0] as i32, xs[1] as i32, xs[2] as i32]),
        _ => Value::Vec4i([xs[0] as i32, xs[1] as i32, xs[2] as i32, xs[3] as i32]),
    })
}

fn matrix_from(n: &Node, dims: usize) -> Result<Value> {
    let rows = match n {
        Node::Tuple(rows) if rows.len() == dims => rows,
        other => {
            return Err(UsdError::Parse {
                line: 0,
                message: format!("matrix{dims} expected, found {other:?}"),
            })
        }
    };
    let mut flat = Vec::with_capacity(dims * dims);
    for r in rows {
        match r {
            Node::Tuple(cells) if cells.len() == dims => {
                for c in cells {
                    flat.push(num(c)?);
                }
            }
            other => {
                return Err(UsdError::Parse {
                    line: 0,
                    message: format!("matrix row {other:?} has the wrong size"),
                })
            }
        }
    }
    Ok(match dims {
        2 => Value::Matrix2d(Box::new([flat[0], flat[1], flat[2], flat[3]])),
        3 => {
            let mut a = [0.0; 9];
            a.copy_from_slice(&flat);
            Value::Matrix3d(Box::new(a))
        }
        _ => {
            let mut a = [0.0; 16];
            a.copy_from_slice(&flat);
            Value::Matrix4d(Box::new(a))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_kind_names() {
        assert_eq!(vector_kind("float3"), Some((3, 'f')));
        assert_eq!(vector_kind("point3f"), Some((3, 'f')));
        assert_eq!(vector_kind("double2"), Some((2, 'd')));
        assert_eq!(vector_kind("color3f"), Some((3, 'f')));
        assert_eq!(vector_kind("quatf"), None);
        assert_eq!(vector_kind("token"), None);
    }

    #[test]
    fn lexer_handles_numbers_and_samples() {
        let toks = lex("float x.timeSamples = { 0: -1.5e-3, 1: 2 }").unwrap();
        assert!(toks
            .iter()
            .any(|t| t.tok == Tok::Ident("x.timeSamples".into())));
        assert!(toks.iter().any(|t| t.tok == Tok::Num("-1.5e-3".into())));
    }

    #[test]
    fn small_document_round_trips() {
        let text = r#"#usda 1.0
(
    token defaultPrim = "Robot"
)

def Xform "Robot" (
    apiSchemas = ["PhysicsArticulationRootAPI"]
) {
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double3 xformOp:translate = (0.5, -1, 2e-3)
    float mass = 3.5
    rel physics:body0 = </Robot/pelvis>
    float3[] points = [(0, 0, 0), (1, 2, 3)]
    float angle.timeSamples = { 0: 1.0, 2: 3.0 }
    def Sphere "ball" {
        double radius = 0.25
    }
}
"#;
        let stage = parse_stage(text).expect("parse");
        let robot = stage.prim("/Robot").expect("prim");
        assert_eq!(robot.type_name.as_deref(), Some("Xform"));
        assert_eq!(
            robot.api_schemas,
            vec!["PhysicsArticulationRootAPI".to_string()]
        );
        let mass = robot.property("mass").unwrap();
        assert_eq!(mass.default, Some(Value::Float(3.5)));
        let pts = robot.property("points").unwrap();
        assert_eq!(
            pts.default,
            Some(Value::Array(vec![
                Value::Vec3f([0.0, 0.0, 0.0]),
                Value::Vec3f([1.0, 2.0, 3.0])
            ]))
        );
        assert_eq!(
            robot.property("physics:body0").unwrap().targets,
            vec!["/Robot/pelvis".to_string()]
        );
        assert_eq!(
            robot
                .property("angle")
                .unwrap()
                .time_samples
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            robot.child("ball").unwrap().type_name.as_deref(),
            Some("Sphere")
        );
        assert_eq!(
            stage.metadata.get("defaultPrim"),
            Some(&Value::Token("Robot".into()))
        );

        // Writing the stage and reading it back gives the same prims.
        let again = parse_stage(&write_usda(&stage)).expect("reparse");
        assert_eq!(again.prims, stage.prims);
        assert_eq!(again.metadata, stage.metadata);
    }

    #[test]
    fn parse_errors_carry_line_numbers() {
        match parse_layer("#usda 1.0\n\ndef Xform \"A\" {\n    float x = \n}\n", "t") {
            Err(UsdError::Parse { line, .. }) => assert!(line >= 4, "line {line}"),
            other => panic!("expected parse error, got {other:?}"),
        }
    }
}
