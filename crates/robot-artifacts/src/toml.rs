//! A bounded TOML subset for manifests.
//!
//! SPECS.md Section 7.5 makes TOML the manifest format and Section 2 states that
//! "TOML configuration, schemas, Markdown documentation, and data files are not
//! programming-language exceptions". Section 13 requires treating input as
//! untrusted: "enforce size limits, reject malformed dimensions, fuzz parsers".
//!
//! This parser therefore implements a *documented subset* and rejects everything
//! else explicitly, which is the same discipline SPECS.md Section 7.1 demands of
//! URDF import. The subset is:
//!
//! * key/value pairs with bare keys (`[A-Za-z0-9_-]+`) and dotted keys;
//! * basic strings with the escapes `\" \\ \n \r \t \0`, no multi-line strings,
//!   no literal strings;
//! * integers (decimal, optional sign, optional `_` separators), floats
//!   (including `inf` and `nan` so a report can round-trip one), booleans;
//! * inline arrays of any supported value, including nested arrays;
//! * `[table]` and `[[array.of.tables]]` headers;
//! * `#` comments to end of line.
//!
//! Dates, times, inline tables, multi-line strings, literal strings, hex/octal/
//! binary integers and unicode escapes are **not** supported and produce
//! [`TomlError::Unsupported`] naming the construct, rather than being skipped.

use core::fmt;

/// Hard limits applied while parsing untrusted input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TomlLimits {
    /// Maximum input length in bytes.
    pub max_bytes: usize,
    /// Maximum nesting depth of arrays.
    pub max_depth: usize,
    /// Maximum number of entries across the whole document.
    pub max_entries: usize,
    /// Maximum length of a single string value.
    pub max_string_len: usize,
    /// Maximum number of elements in one array.
    pub max_array_len: usize,
}

impl Default for TomlLimits {
    /// Limits sized for a manifest, not for a corpus.
    ///
    /// A manifest that needs more than this is carrying data that belongs in a
    /// content-addressed blob (SPECS.md Section 7.5).
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024,
            max_depth: 16,
            max_entries: 100_000,
            max_string_len: 64 * 1024,
            max_array_len: 100_000,
        }
    }
}

/// A value in the supported subset.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    /// A boolean.
    Boolean(bool),
    /// A 64-bit signed integer.
    Integer(i64),
    /// A double.
    Float(f64),
    /// A basic string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// A table, with entries in insertion order.
    Table(Table),
}

impl Value {
    /// The subset's name for this value's type, used in error messages.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Boolean(_) => "boolean",
            Self::Integer(_) => "integer",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Table(_) => "table",
        }
    }

    /// The value as a boolean.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// The value as an integer.
    #[must_use]
    pub const fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// The value as a float, accepting an integer losslessly.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// The value as a string slice.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The value as a table.
    #[must_use]
    pub const fn as_table(&self) -> Option<&Table> {
        match self {
            Self::Table(t) => Some(t),
            _ => None,
        }
    }

    /// An array of floats, accepting integers.
    #[must_use]
    pub fn as_float_array(&self) -> Option<Vec<f64>> {
        self.as_array()?.iter().map(Value::as_float).collect()
    }
}

/// An ordered table.
///
/// Insertion order is preserved so that writing a table back produces the same
/// bytes it was read from, which is what makes a manifest's digest stable across
/// a tooling round trip (SPECS.md Section 7.1, preservation of unknown
/// extensions).
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Table {
    entries: Vec<(String, Value)>,
}

impl Table {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a key.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        let key = key.into();
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => Some(core::mem::replace(&mut slot.1, value)),
            None => {
                self.entries.push((key, value));
                None
            }
        }
    }

    /// Looks up a key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Looks up a dotted path.
    #[must_use]
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        let mut current = self;
        let mut parts = path.split('.').peekable();
        while let Some(part) = parts.next() {
            let value = current.get(part)?;
            if parts.peek().is_none() {
                return Some(value);
            }
            current = value.as_table()?;
        }
        None
    }

    /// Entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the table has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A required value, or a named error.
    pub fn require(&self, key: &str) -> Result<&Value, TomlError> {
        self.get(key).ok_or_else(|| TomlError::MissingKey {
            key: key.to_owned(),
        })
    }

    /// A required string.
    pub fn require_str(&self, key: &str) -> Result<&str, TomlError> {
        let v = self.require(key)?;
        v.as_str().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "string",
            found: v.type_name(),
        })
    }

    /// A required integer.
    pub fn require_integer(&self, key: &str) -> Result<i64, TomlError> {
        let v = self.require(key)?;
        v.as_integer().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "integer",
            found: v.type_name(),
        })
    }

    /// A required float, accepting an integer.
    pub fn require_float(&self, key: &str) -> Result<f64, TomlError> {
        let v = self.require(key)?;
        v.as_float().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "float",
            found: v.type_name(),
        })
    }

    /// A required boolean.
    pub fn require_bool(&self, key: &str) -> Result<bool, TomlError> {
        let v = self.require(key)?;
        v.as_bool().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "boolean",
            found: v.type_name(),
        })
    }

    /// A required table.
    pub fn require_table(&self, key: &str) -> Result<&Table, TomlError> {
        let v = self.require(key)?;
        v.as_table().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "table",
            found: v.type_name(),
        })
    }

    /// A required array.
    pub fn require_array(&self, key: &str) -> Result<&[Value], TomlError> {
        let v = self.require(key)?;
        v.as_array().ok_or_else(|| TomlError::WrongType {
            key: key.to_owned(),
            expected: "array",
            found: v.type_name(),
        })
    }

    /// Keys present in this table that are not in `known`.
    ///
    /// SPECS.md Section 7.1 requires unknown core fields to be rejected before
    /// activation; this is the check that finds them.
    #[must_use]
    pub fn unknown_keys(&self, known: &[&str]) -> Vec<String> {
        self.entries
            .iter()
            .map(|(k, _)| k)
            .filter(|k| !known.contains(&k.as_str()))
            .cloned()
            .collect()
    }
}

/// Why parsing failed.
#[derive(Clone, PartialEq, Debug)]
pub enum TomlError {
    /// A construct outside the supported subset was present.
    Unsupported {
        /// The construct's name.
        construct: &'static str,
        /// Byte offset where it began.
        offset: usize,
    },
    /// The input did not parse.
    Syntax {
        /// What was expected.
        expected: &'static str,
        /// Byte offset.
        offset: usize,
    },
    /// A limit from [`TomlLimits`] was exceeded.
    LimitExceeded {
        /// Which limit.
        limit: &'static str,
        /// The limit's value.
        value: usize,
    },
    /// A key was defined twice.
    DuplicateKey {
        /// The key.
        key: String,
        /// Byte offset of the second definition.
        offset: usize,
    },
    /// A required key was absent.
    MissingKey {
        /// The key.
        key: String,
    },
    /// A key held the wrong type.
    WrongType {
        /// The key.
        key: String,
        /// Type required.
        expected: &'static str,
        /// Type found.
        found: &'static str,
    },
    /// A table header named a path that is already a non-table value.
    PathConflict {
        /// The path.
        path: String,
    },
}

impl fmt::Display for TomlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { construct, offset } => write!(
                f,
                "{construct} at byte {offset} is outside the supported TOML subset"
            ),
            Self::Syntax { expected, offset } => {
                write!(f, "expected {expected} at byte {offset}")
            }
            Self::LimitExceeded { limit, value } => {
                write!(f, "exceeded {limit} limit of {value}")
            }
            Self::DuplicateKey { key, offset } => {
                write!(f, "key {key:?} redefined at byte {offset}")
            }
            Self::MissingKey { key } => write!(f, "missing required key {key:?}"),
            Self::WrongType {
                key,
                expected,
                found,
            } => {
                write!(f, "key {key:?} is a {found}, expected a {expected}")
            }
            Self::PathConflict { path } => {
                write!(f, "table path {path:?} conflicts with an existing value")
            }
        }
    }
}

impl core::error::Error for TomlError {}

/// Parses a document with the default limits.
pub fn parse(input: &str) -> Result<Table, TomlError> {
    parse_with_limits(input, TomlLimits::default())
}

/// Parses a document with explicit limits.
pub fn parse_with_limits(input: &str, limits: TomlLimits) -> Result<Table, TomlError> {
    if input.len() > limits.max_bytes {
        return Err(TomlError::LimitExceeded {
            limit: "max_bytes",
            value: limits.max_bytes,
        });
    }
    Parser {
        bytes: input.as_bytes(),
        pos: 0,
        limits,
        entries: 0,
    }
    .document()
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: TomlLimits,
    entries: usize,
}

impl<'a> Parser<'a> {
    fn document(&mut self) -> Result<Table, TomlError> {
        let mut root = Table::new();
        // Path of the table currently being filled, empty for the root.
        let mut current: Vec<String> = Vec::new();
        loop {
            self.skip_trivia();
            if self.pos >= self.bytes.len() {
                break;
            }
            if self.peek() == Some(b'[') {
                let (path, array) = self.table_header()?;
                if array {
                    append_array_table(&mut root, &path)?;
                } else {
                    ensure_table(&mut root, &path)?;
                }
                current = path;
            } else {
                let (key_path, value) = self.key_value()?;
                self.count_entry()?;
                let mut full = current.clone();
                full.extend(key_path);
                insert_path(&mut root, &full, value, self.pos)?;
            }
        }
        Ok(root)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn count_entry(&mut self) -> Result<(), TomlError> {
        self.entries += 1;
        if self.entries > self.limits.max_entries {
            return Err(TomlError::LimitExceeded {
                limit: "max_entries",
                value: self.limits.max_entries,
            });
        }
        Ok(())
    }

    /// Skips whitespace, newlines and comments.
    fn skip_trivia(&mut self) {
        while let Some(c) = self.peek() {
            match c {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                b'#' => {
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    /// Skips spaces and tabs only.
    fn skip_inline_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t')) {
            self.pos += 1;
        }
    }

    /// Reads `[path]` or `[[path]]`, returning the path and whether it was an
    /// array-of-tables header.
    fn table_header(&mut self) -> Result<(Vec<String>, bool), TomlError> {
        debug_assert_eq!(self.peek(), Some(b'['));
        self.pos += 1;
        let array = self.peek() == Some(b'[');
        if array {
            self.pos += 1;
        }
        let mut path = Vec::new();
        loop {
            self.skip_inline_space();
            path.push(self.bare_key()?);
            self.skip_inline_space();
            match self.peek() {
                Some(b'.') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    if array {
                        if self.peek() != Some(b']') {
                            return Err(TomlError::Syntax {
                                expected: "]]",
                                offset: self.pos,
                            });
                        }
                        self.pos += 1;
                    }
                    break;
                }
                _ => {
                    return Err(TomlError::Syntax {
                        expected: "'.' or ']'",
                        offset: self.pos,
                    })
                }
            }
        }
        Ok((path, array))
    }

    fn bare_key(&mut self) -> Result<String, TomlError> {
        if self.peek() == Some(b'"') {
            return Err(TomlError::Unsupported {
                construct: "quoted key",
                offset: self.pos,
            });
        }
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(TomlError::Syntax {
                expected: "bare key",
                offset: self.pos,
            });
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn key_value(&mut self) -> Result<(Vec<String>, Value), TomlError> {
        let mut path = vec![self.bare_key()?];
        loop {
            self.skip_inline_space();
            match self.peek() {
                Some(b'.') => {
                    self.pos += 1;
                    self.skip_inline_space();
                    path.push(self.bare_key()?);
                }
                Some(b'=') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(TomlError::Syntax {
                        expected: "'=' or '.'",
                        offset: self.pos,
                    })
                }
            }
        }
        self.skip_inline_space();
        let value = self.value(0)?;
        // Only trivia may follow a value on its line.
        self.skip_inline_space();
        match self.peek() {
            None | Some(b'\n') | Some(b'\r') | Some(b'#') => Ok((path, value)),
            _ => Err(TomlError::Syntax {
                expected: "end of line",
                offset: self.pos,
            }),
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, TomlError> {
        if depth > self.limits.max_depth {
            return Err(TomlError::LimitExceeded {
                limit: "max_depth",
                value: self.limits.max_depth,
            });
        }
        match self.peek() {
            Some(b'"') => self.basic_string().map(Value::String),
            Some(b'\'') => Err(TomlError::Unsupported {
                construct: "literal string",
                offset: self.pos,
            }),
            Some(b'{') => Err(TomlError::Unsupported {
                construct: "inline table",
                offset: self.pos,
            }),
            Some(b'[') => self.array(depth),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(c) if c == b'+' || c == b'-' || c.is_ascii_digit() || c == b'i' || c == b'n' => {
                self.number()
            }
            _ => Err(TomlError::Syntax {
                expected: "value",
                offset: self.pos,
            }),
        }
    }

    fn basic_string(&mut self) -> Result<String, TomlError> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        if self.bytes[self.pos..].starts_with(b"\"\"\"") {
            return Err(TomlError::Unsupported {
                construct: "multi-line string",
                offset: self.pos,
            });
        }
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or(TomlError::Syntax {
                expected: "closing quote",
                offset: self.pos,
            })?;
            self.pos += 1;
            match c {
                b'"' => break,
                b'\n' => {
                    return Err(TomlError::Syntax {
                        expected: "closing quote before newline",
                        offset: self.pos,
                    })
                }
                b'\\' => {
                    let esc = self.peek().ok_or(TomlError::Syntax {
                        expected: "escape character",
                        offset: self.pos,
                    })?;
                    self.pos += 1;
                    out.push(match esc {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'0' => '\0',
                        b'u' | b'U' => {
                            return Err(TomlError::Unsupported {
                                construct: "unicode escape",
                                offset: self.pos - 1,
                            })
                        }
                        _ => {
                            return Err(TomlError::Syntax {
                                expected: "a supported escape",
                                offset: self.pos - 1,
                            })
                        }
                    });
                }
                _ => {
                    // Re-decode the UTF-8 sequence this byte starts.
                    let start = self.pos - 1;
                    let len = utf8_len(c);
                    if start + len > self.bytes.len() {
                        return Err(TomlError::Syntax {
                            expected: "valid UTF-8",
                            offset: start,
                        });
                    }
                    let s =
                        core::str::from_utf8(&self.bytes[start..start + len]).map_err(|_| {
                            TomlError::Syntax {
                                expected: "valid UTF-8",
                                offset: start,
                            }
                        })?;
                    out.push_str(s);
                    self.pos = start + len;
                }
            }
            if out.len() > self.limits.max_string_len {
                return Err(TomlError::LimitExceeded {
                    limit: "max_string_len",
                    value: self.limits.max_string_len,
                });
            }
        }
        Ok(out)
    }

    fn boolean(&mut self) -> Result<Value, TomlError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Boolean(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Boolean(false))
        } else {
            Err(TomlError::Syntax {
                expected: "true or false",
                offset: self.pos,
            })
        }
    }

    fn number(&mut self) -> Result<Value, TomlError> {
        let start = self.pos;
        if self.bytes[self.pos..].starts_with(b"0x")
            || self.bytes[self.pos..].starts_with(b"0o")
            || self.bytes[self.pos..].starts_with(b"0b")
        {
            return Err(TomlError::Unsupported {
                construct: "non-decimal integer",
                offset: self.pos,
            });
        }
        let mut text = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => {
                    text.push(c as char);
                    self.pos += 1;
                }
                b'+' | b'-' => {
                    // A sign may lead the number or follow an exponent marker.
                    // Anywhere else it is a date separator, which this subset
                    // rejects by name rather than misparsing.
                    let after_exponent = matches!(text.chars().last(), Some('e') | Some('E'));
                    if !(text.is_empty() || after_exponent) {
                        return Err(TomlError::Unsupported {
                            construct: "date or time",
                            offset: start,
                        });
                    }
                    text.push(c as char);
                    self.pos += 1;
                }
                b'_' => self.pos += 1,
                b'.' | b'e' | b'E' => {
                    is_float = true;
                    text.push(c as char);
                    self.pos += 1;
                }
                b'i' | b'n' | b'f' | b'a' => {
                    is_float = true;
                    text.push(c as char);
                    self.pos += 1;
                }
                _ => break,
            }
        }
        if text.is_empty() {
            return Err(TomlError::Syntax {
                expected: "number",
                offset: start,
            });
        }
        if is_float {
            let normalized = match text.as_str() {
                "inf" | "+inf" => "inf".to_owned(),
                "-inf" => "-inf".to_owned(),
                "nan" | "+nan" | "-nan" => "NaN".to_owned(),
                other => other.to_owned(),
            };
            normalized
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| TomlError::Syntax {
                    expected: "float",
                    offset: start,
                })
        } else {
            text.parse::<i64>()
                .map(Value::Integer)
                .map_err(|_| TomlError::Syntax {
                    expected: "64-bit integer",
                    offset: start,
                })
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, TomlError> {
        debug_assert_eq!(self.peek(), Some(b'['));
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_trivia();
            match self.peek() {
                None => {
                    return Err(TomlError::Syntax {
                        expected: "']'",
                        offset: self.pos,
                    })
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {}
            }
            items.push(self.value(depth + 1)?);
            if items.len() > self.limits.max_array_len {
                return Err(TomlError::LimitExceeded {
                    limit: "max_array_len",
                    value: self.limits.max_array_len,
                });
            }
            self.skip_trivia();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(TomlError::Syntax {
                        expected: "',' or ']'",
                        offset: self.pos,
                    })
                }
            }
        }
        Ok(Value::Array(items))
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

/// Ensures `path` names a table, creating intermediate tables.
fn ensure_table<'t>(root: &'t mut Table, path: &[String]) -> Result<&'t mut Table, TomlError> {
    let mut current = root;
    for (i, key) in path.iter().enumerate() {
        let is_present = current.get(key).is_some();
        if !is_present {
            current.insert(key.clone(), Value::Table(Table::new()));
        }
        let slot = current
            .entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .expect("just inserted");
        current = match slot {
            Value::Table(t) => t,
            // A later `[a.b]` after `[[a]]` applies to the last element.
            Value::Array(items) => match items.last_mut() {
                Some(Value::Table(t)) => t,
                _ => {
                    return Err(TomlError::PathConflict {
                        path: path[..=i].join("."),
                    });
                }
            },
            _ => {
                return Err(TomlError::PathConflict {
                    path: path[..=i].join("."),
                })
            }
        };
    }
    Ok(current)
}

/// Appends a new table to the array of tables at `path`.
fn append_array_table(root: &mut Table, path: &[String]) -> Result<(), TomlError> {
    let (last, parents) = path.split_last().ok_or(TomlError::Syntax {
        expected: "table path",
        offset: 0,
    })?;
    let parent = ensure_table(root, parents)?;
    match parent.get(last) {
        None => {
            parent.insert(last.clone(), Value::Array(vec![Value::Table(Table::new())]));
        }
        Some(Value::Array(_)) => {
            let slot = parent
                .entries
                .iter_mut()
                .find(|(k, _)| k == last)
                .map(|(_, v)| v)
                .expect("present");
            if let Value::Array(items) = slot {
                items.push(Value::Table(Table::new()));
            }
        }
        Some(_) => {
            return Err(TomlError::PathConflict {
                path: path.join("."),
            })
        }
    }
    Ok(())
}

/// Inserts a value at a dotted path, rejecting a redefinition.
fn insert_path(
    root: &mut Table,
    path: &[String],
    value: Value,
    offset: usize,
) -> Result<(), TomlError> {
    let (last, parents) = path.split_last().ok_or(TomlError::Syntax {
        expected: "key",
        offset,
    })?;
    let parent = ensure_table(root, parents)?;
    if parent.get(last).is_some() {
        return Err(TomlError::DuplicateKey {
            key: path.join("."),
            offset,
        });
    }
    parent.insert(last.clone(), value);
    Ok(())
}

/// Serializes a table back to TOML.
///
/// The output parses back to an equal table, which the round-trip test asserts.
/// Scalars are written before sub-tables so that the result is valid TOML
/// regardless of insertion order.
#[must_use]
pub fn to_string(table: &Table) -> String {
    let mut out = String::new();
    write_table(&mut out, table, &[]);
    out
}

/// True when a value is written as an `[[array of tables]]` rather than inline.
fn is_table_array(value: &Value) -> bool {
    match value {
        Value::Array(items) => {
            !items.is_empty() && items.iter().all(|i| matches!(i, Value::Table(_)))
        }
        _ => false,
    }
}

fn write_table(out: &mut String, table: &Table, path: &[&str]) {
    // Inline values first, in their original order, so that a reparse yields the
    // same insertion order and therefore an equal `Table`.
    for (key, value) in table.iter() {
        if matches!(value, Value::Table(_)) || is_table_array(value) {
            continue;
        }
        out.push_str(key);
        out.push_str(" = ");
        write_value(out, value);
        out.push('\n');
    }
    for (key, value) in table.iter() {
        match value {
            Value::Table(sub) => {
                let mut sub_path = path.to_vec();
                sub_path.push(key);
                out.push('\n');
                out.push('[');
                out.push_str(&sub_path.join("."));
                out.push_str("]\n");
                write_table(out, sub, &sub_path);
            }
            Value::Array(items) if is_table_array(value) => {
                let mut sub_path = path.to_vec();
                sub_path.push(key);
                for item in items {
                    if let Value::Table(sub) = item {
                        out.push('\n');
                        out.push_str("[[");
                        out.push_str(&sub_path.join("."));
                        out.push_str("]]\n");
                        write_table(out, sub, &sub_path);
                    }
                }
            }
            _ => {}
        }
    }
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Integer(i) => out.push_str(&i.to_string()),
        Value::Float(v) => {
            if v.is_nan() {
                out.push_str("nan");
            } else if v.is_infinite() {
                out.push_str(if *v > 0.0 { "inf" } else { "-inf" });
            } else if v.fract() == 0.0 && v.abs() < 1e15 {
                // Keep the decimal point so the value reparses as a float.
                out.push_str(&format!("{v:.1}"));
            } else {
                out.push_str(&format!("{v:?}"));
            }
        }
        Value::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\0' => out.push_str("\\0"),
                    _ => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Table(_) => {
            // Reached only for a table nested inside an array, which the subset
            // does not produce; emitted as an empty inline table rather than
            // silently dropping it.
            out.push_str("{}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_and_tables() {
        let doc = r#"
# a comment
schema_version = "1.0"
count = 42
rate_hz = 100.0
enabled = true
tags = ["jab", "guard"]

[envelope]
torque_limit = 10.5
position_limits = [[-1.0, 1.0], [-2.0, 2.0]]
"#;
        let t = parse(doc).unwrap();
        assert_eq!(t.require_str("schema_version").unwrap(), "1.0");
        assert_eq!(t.require_integer("count").unwrap(), 42);
        assert!((t.require_float("rate_hz").unwrap() - 100.0).abs() < 1e-12);
        assert!(t.require_bool("enabled").unwrap());
        assert_eq!(t.require_array("tags").unwrap().len(), 2);
        assert!(
            (t.get_path("envelope.torque_limit")
                .unwrap()
                .as_float()
                .unwrap()
                - 10.5)
                .abs()
                < 1e-12
        );
        let limits = t
            .get_path("envelope.position_limits")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(limits.len(), 2);
        assert_eq!(limits[0].as_float_array().unwrap(), vec![-1.0, 1.0]);
    }

    #[test]
    fn parses_arrays_of_tables() {
        let doc = r#"
[[component]]
name = "actor"
tier = "policy"

[[component]]
name = "strategy"
tier = "tactical"
"#;
        let t = parse(doc).unwrap();
        let components = t.require_array("component").unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(
            components[0]
                .as_table()
                .unwrap()
                .require_str("name")
                .unwrap(),
            "actor"
        );
        assert_eq!(
            components[1]
                .as_table()
                .unwrap()
                .require_str("tier")
                .unwrap(),
            "tactical"
        );
    }

    #[test]
    fn parses_dotted_keys() {
        let t = parse("a.b.c = 1\na.b.d = 2\n").unwrap();
        assert_eq!(t.get_path("a.b.c").unwrap().as_integer(), Some(1));
        assert_eq!(t.get_path("a.b.d").unwrap().as_integer(), Some(2));
    }

    #[test]
    fn round_trips_through_text() {
        let doc = r#"
name = "tracker"
rate = 100
scale = [1.0, 0.5, -2.25]
flag = false

[lineage]
source = "git:abc"
inputs = ["sha256:a", "sha256:b"]

[[benchmark]]
metric = "tracking_error"
value = 0.031
"#;
        let parsed = parse(doc).unwrap();
        let written = to_string(&parsed);
        let reparsed = parse(&written).unwrap();
        assert_eq!(
            parsed, reparsed,
            "round trip changed the document:\n{written}"
        );
    }

    #[test]
    fn special_floats_round_trip() {
        let t = parse("a = inf\nb = -inf\nc = nan\n").unwrap();
        assert!(t.require_float("a").unwrap().is_infinite());
        assert!(t.require_float("b").unwrap() < 0.0);
        assert!(t.require_float("c").unwrap().is_nan());
        let written = to_string(&t);
        let back = parse(&written).unwrap();
        assert!(back.require_float("c").unwrap().is_nan());
    }

    #[test]
    fn escapes_round_trip() {
        let t = parse("s = \"line\\nquote\\\"tab\\tdone\"\n").unwrap();
        assert_eq!(t.require_str("s").unwrap(), "line\nquote\"tab\tdone");
        let back = parse(&to_string(&t)).unwrap();
        assert_eq!(back.require_str("s").unwrap(), "line\nquote\"tab\tdone");
    }

    #[test]
    fn non_ascii_strings_survive() {
        let t = parse("s = \"caf\u{e9} \u{1f916}\"\n").unwrap();
        assert_eq!(t.require_str("s").unwrap(), "caf\u{e9} \u{1f916}");
        assert_eq!(parse(&to_string(&t)).unwrap(), t);
    }

    #[test]
    fn unsupported_constructs_are_named_not_skipped() {
        for (doc, construct) in [
            ("a = 'literal'\n", "literal string"),
            ("a = {b = 1}\n", "inline table"),
            ("a = \"\"\"multi\"\"\"\n", "multi-line string"),
            ("a = 0x1f\n", "non-decimal integer"),
            ("a = \"\\u00e9\"\n", "unicode escape"),
            ("\"quoted\" = 1\n", "quoted key"),
        ] {
            match parse(doc) {
                Err(TomlError::Unsupported { construct: c, .. }) => assert_eq!(c, construct),
                other => panic!("expected {construct:?} to be rejected, got {other:?}"),
            }
        }
    }

    #[test]
    fn dates_are_rejected_by_name() {
        assert!(matches!(
            parse("d = 2026-10-26\n"),
            Err(TomlError::Unsupported {
                construct: "date or time",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        assert!(matches!(
            parse("a = 1\na = 2\n"),
            Err(TomlError::DuplicateKey { .. })
        ));
    }

    #[test]
    fn limits_are_enforced() {
        let tiny = TomlLimits {
            max_bytes: 4,
            ..TomlLimits::default()
        };
        assert!(matches!(
            parse_with_limits("a = 12345\n", tiny),
            Err(TomlError::LimitExceeded {
                limit: "max_bytes",
                ..
            })
        ));

        let shallow = TomlLimits {
            max_depth: 2,
            ..TomlLimits::default()
        };
        assert!(matches!(
            parse_with_limits("a = [[[[1]]]]\n", shallow),
            Err(TomlError::LimitExceeded {
                limit: "max_depth",
                ..
            })
        ));

        let few = TomlLimits {
            max_entries: 1,
            ..TomlLimits::default()
        };
        assert!(matches!(
            parse_with_limits("a = 1\nb = 2\n", few),
            Err(TomlError::LimitExceeded {
                limit: "max_entries",
                ..
            })
        ));

        let short = TomlLimits {
            max_string_len: 2,
            ..TomlLimits::default()
        };
        assert!(matches!(
            parse_with_limits("a = \"abcdef\"\n", short),
            Err(TomlError::LimitExceeded {
                limit: "max_string_len",
                ..
            })
        ));

        let narrow = TomlLimits {
            max_array_len: 2,
            ..TomlLimits::default()
        };
        assert!(matches!(
            parse_with_limits("a = [1,2,3]\n", narrow),
            Err(TomlError::LimitExceeded {
                limit: "max_array_len",
                ..
            })
        ));
    }

    #[test]
    fn truncated_input_is_rejected_rather_than_accepted() {
        for doc in ["a = \"unterminated", "a = [1, 2", "a =", "[table", "a"] {
            assert!(parse(doc).is_err(), "{doc:?} should not parse");
        }
    }

    #[test]
    fn trailing_junk_after_a_value_is_rejected() {
        assert!(matches!(
            parse("a = 1 garbage\n"),
            Err(TomlError::Syntax {
                expected: "end of line",
                ..
            })
        ));
    }

    #[test]
    fn unknown_keys_are_reported() {
        let t = parse("known = 1\nsurprise = 2\n").unwrap();
        assert_eq!(t.unknown_keys(&["known"]), vec!["surprise".to_owned()]);
        assert!(t.unknown_keys(&["known", "surprise"]).is_empty());
    }

    #[test]
    fn wrong_types_name_both_sides() {
        let t = parse("a = 1\n").unwrap();
        assert_eq!(
            t.require_str("a"),
            Err(TomlError::WrongType {
                key: "a".to_owned(),
                expected: "string",
                found: "integer"
            })
        );
        assert_eq!(
            t.require("b"),
            Err(TomlError::MissingKey {
                key: "b".to_owned()
            })
        );
    }

    #[test]
    fn integer_overflow_is_rejected() {
        assert!(matches!(
            parse("a = 99999999999999999999999\n"),
            Err(TomlError::Syntax {
                expected: "64-bit integer",
                ..
            })
        ));
    }

    #[test]
    fn a_table_path_colliding_with_a_scalar_is_rejected() {
        assert!(matches!(
            parse("a = 1\n[a.b]\nc = 2\n"),
            Err(TomlError::PathConflict { .. })
        ));
    }

    #[test]
    fn parser_survives_adversarial_byte_sequences() {
        // Not coverage-guided fuzzing — SPECS.md Appendix B.1 correction 9 keeps
        // libFuzzer out of the default toolchain — but a deterministic sweep
        // that asserts the parser always terminates with a typed result.
        let seeds: [&str; 10] = [
            "",
            "[",
            "[[",
            "]]",
            "=",
            "\"",
            "\\",
            "a=",
            "[a]\n[a]\n",
            "a=[",
        ];
        let mut rng = robot_types::rng::Rng::new(robot_types::rng::SeedStream::new(7, 0));
        for seed in seeds {
            for _ in 0..200 {
                let mut doc = seed.to_owned();
                for _ in 0..(rng.below(24) as usize) {
                    let c = b"[]{}=\"'.,\n\t 0123456789abcdefinrtu_-+#"[rng.below(37) as usize];
                    doc.push(c as char);
                }
                // The contract is only that this returns rather than hangs,
                // overflows or panics.
                let _ = parse(&doc);
            }
        }
    }
}
