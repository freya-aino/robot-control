//! Layer composition.
//!
//! The root layer's `subLayers` are loaded recursively, strongest first. A
//! prim is then composed from a list of *sources*: `(layer stack, spec path)`
//! pairs. The list starts with the prim's own specs and is extended, in USD's
//! LIVRPS order, by variant selections, then references, then payloads. Each
//! field takes the value of the strongest source that authors it. Child and
//! property orders are unions of the sources' orders, strongest first.
//!
//! Implemented arcs: sublayers (identity layer offsets only), variant selections,
//! internal and external references, and payloads, each with an optional
//! prim path (default prim when empty) and identity layer offset. Relationship
//! targets and connections are kept exactly as authored, not remapped through
//! arcs. Deleted or ordered list items and standalone connection specs are
//! reported as [`UsdError::Unsupported`].

use crate::error::{Result, UsdError};
use crate::layer::{read_layer_file, resolve_relative, Layer, Spec, SpecKind};
use crate::value::{Arc, ListOp, Specifier, Value, Variability};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

/// Whether a property is an attribute or a relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyKind {
    /// An attribute, with a type name and optional default and time samples.
    Attribute,
    /// A relationship, with targets.
    Relationship,
}

/// A composed attribute or relationship.
#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    /// The property name, for example `physics:mass`.
    pub name: String,
    /// The property path, for example `/G1/left_hip.physics:mass`.
    pub path: String,
    /// Attribute or relationship.
    pub kind: PropertyKind,
    /// The `typeName` of an attribute, for example `float3`.
    pub type_name: Option<String>,
    /// The `variability`, if authored.
    pub variability: Option<Variability>,
    /// The `default` value of an attribute, if authored.
    pub default: Option<Value>,
    /// The `timeSamples` of an attribute, if authored.
    pub time_samples: Option<Vec<(f64, Value)>>,
    /// Connection paths of an attribute (`connectionPaths`).
    pub connections: Vec<String>,
    /// Target paths of a relationship (`targetPaths`).
    pub targets: Vec<String>,
    /// Any other authored fields, in stored order.
    pub metadata: BTreeMap<String, Value>,
}

/// A composed prim.
#[derive(Debug, Clone, PartialEq)]
pub struct Prim {
    /// The prim name (the last path element).
    pub name: String,
    /// The prim path, for example `/G1/left_hip`.
    pub path: String,
    /// `def`, `over` or `class`, if authored.
    pub specifier: Option<Specifier>,
    /// The `typeName`, for example `Xform` or `Cube`.
    pub type_name: Option<String>,
    /// The `kind` metadata, if authored.
    pub kind: Option<String>,
    /// The `apiSchemas` list, effective items.
    pub api_schemas: Vec<String>,
    /// Properties in composed order.
    pub properties: Vec<Property>,
    /// Child prims in composed order.
    pub children: Vec<Prim>,
    /// Any other authored fields, in stored order.
    pub metadata: BTreeMap<String, Value>,
}

impl Prim {
    /// Returns the property with this name, if any.
    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// Returns the child with this name, if any.
    pub fn child(&self, name: &str) -> Option<&Prim> {
        self.children.iter().find(|p| p.name == name)
    }
}

/// A composed stage: the layer stack (strongest first), root metadata and the
/// top-level prims.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// Identifiers of the root layer and its sublayers, strongest first.
    pub layers: Vec<String>,
    /// Pseudo-root metadata (for example `defaultPrim`, `upAxis`,
    /// `metersPerUnit`), excluding `subLayers`.
    pub metadata: BTreeMap<String, Value>,
    /// Top-level prims in composed order.
    pub prims: Vec<Prim>,
}

impl Stage {
    /// Loads `root`, its sublayers and every arc target, and composes them.
    pub fn compose(root: &Path) -> Result<Stage> {
        let id = root.display().to_string();
        let mut c = Composer::default();
        let root_idx = c.load(&id)?;
        let stack = c.stack_of(root_idx)?;
        c.compose_root(stack)
    }

    /// Composes already-loaded layers, strongest first. The first layer is the
    /// root. Sublayers and arc targets that name files are read from disk,
    /// relative to the first layer's identifier.
    pub fn from_layers(layers: Vec<Layer>) -> Result<Stage> {
        let mut c = Composer::default();
        let mut idxs = Vec::new();
        for l in layers {
            check_layer(&l)?;
            idxs.push(c.add(l));
        }
        let first = *idxs
            .first()
            .ok_or_else(|| UsdError::Malformed("no layers to compose".into()))?;
        let stack = c.stack_of(first)?;
        c.compose_root(stack)
    }

    /// Finds a prim by its path, for example `/G1/left_hip`.
    pub fn prim(&self, path: &str) -> Option<&Prim> {
        let mut cur: &[Prim] = &self.prims;
        let mut found = None;
        for name in path.split('/').filter(|s| !s.is_empty()) {
            let p = cur.iter().find(|p| p.name == name)?;
            found = Some(p);
            cur = &p.children;
        }
        found
    }

    /// Calls `f` on every prim, depth-first, in composed order.
    pub fn for_each_prim<'a>(&'a self, mut f: impl FnMut(&'a Prim)) {
        fn walk<'a>(prims: &'a [Prim], f: &mut impl FnMut(&'a Prim)) {
            for p in prims {
                f(p);
                walk(&p.children, f);
            }
        }
        walk(&self.prims, &mut f);
    }

    /// Counts prims by `typeName`. Prims without a type are counted under `""`.
    pub fn count_by_type(&self) -> BTreeMap<String, usize> {
        let mut m = BTreeMap::new();
        self.for_each_prim(|p| {
            *m.entry(p.type_name.clone().unwrap_or_default())
                .or_insert(0) += 1;
        });
        m
    }
}

/// A place a prim's opinions can come from: a spec path in a layer stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Src {
    stack: usize,
    path: String,
}

#[derive(Default)]
struct Composer {
    layers: Vec<Layer>,
    /// Per layer: spec path to spec index.
    index: Vec<HashMap<String, usize>>,
    by_id: HashMap<String, usize>,
    /// Each stack: layer indices, strongest first.
    stacks: Vec<Vec<usize>>,
    stack_of_layer: HashMap<usize, usize>,
    in_progress: HashSet<usize>,
}

const MAX_NESTING: usize = 64;

impl Composer {
    fn add(&mut self, layer: Layer) -> usize {
        let idx = self.layers.len();
        let mut map = HashMap::new();
        for (k, s) in layer.specs.iter().enumerate() {
            map.entry(s.path.clone()).or_insert(k);
        }
        self.by_id.insert(layer.identifier.clone(), idx);
        self.index.push(map);
        self.layers.push(layer);
        idx
    }

    /// Loads a layer by identifier (file path), once.
    fn load(&mut self, id: &str) -> Result<usize> {
        let key = std::fs::canonicalize(id)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| id.to_owned());
        if let Some(&i) = self.by_id.get(&key) {
            return Ok(i);
        }
        let mut layer = read_layer_file(Path::new(id))?;
        layer.identifier = key.clone();
        check_layer(&layer)?;
        Ok(self.add(layer))
    }

    /// The layer stack of a layer: the layer, then its sublayers' stacks.
    fn stack_of(&mut self, layer: usize) -> Result<usize> {
        if let Some(&s) = self.stack_of_layer.get(&layer) {
            return Ok(s);
        }
        if !self.in_progress.insert(layer) {
            return Err(UsdError::Malformed(format!(
                "sublayer cycle through {}",
                self.layers[layer].identifier
            )));
        }
        let mut members = vec![layer];
        let ident = self.layers[layer].identifier.clone();
        for sub in sublayer_refs(&self.layers[layer]) {
            let file = resolve_relative(&ident, &sub).display().to_string();
            let li = self.load(&file)?;
            let st = self.stack_of(li)?;
            for m in self.stacks[st].clone() {
                if !members.contains(&m) {
                    members.push(m);
                }
            }
        }
        self.in_progress.remove(&layer);
        self.stacks.push(members);
        let id = self.stacks.len() - 1;
        self.stack_of_layer.insert(layer, id);
        Ok(id)
    }

    /// The specs at a source, strongest layer first, with their layer index.
    fn specs_at(&self, src: &Src) -> Vec<(usize, usize)> {
        self.stacks[src.stack]
            .iter()
            .filter_map(|&li| self.index[li].get(&src.path).map(|&k| (li, k)))
            .collect()
    }

    fn spec(&self, at: (usize, usize)) -> &Spec {
        &self.layers[at.0].specs[at.1]
    }

    /// Expands variant selections, references and payloads into an ordered
    /// source list: local sources, then variants, then references, then payloads.
    fn gather(&mut self, sources: &[Src], depth: usize) -> Result<Vec<Src>> {
        if depth > MAX_NESTING {
            let at: Vec<String> = sources
                .iter()
                .take(3)
                .map(|s| format!("{}:{}", s.stack, s.path))
                .collect();
            return Err(UsdError::Unsupported(format!(
                "composition arcs nested deeper than 64 (cycle?) at {at:?}"
            )));
        }
        let mut variants = Vec::new();
        let mut refs = Vec::new();
        let mut payloads = Vec::new();
        for s in sources {
            for at in self.specs_at(s) {
                let spec = self.spec(at).clone();
                let layer_id = self.layers[at.0].identifier.clone();
                if let Some(Value::Dictionary(sel)) = spec.field("variantSelection") {
                    let mut names = Vec::new();
                    if let Some(Value::ListOp(op)) = spec.field("variantSetNames") {
                        names = op
                            .effective_items()
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_owned))
                            .collect();
                    }
                    if names.is_empty() {
                        names = sel.iter().map(|(k, _)| k.clone()).collect();
                    }
                    for set in names {
                        if let Some((_, Value::String(choice))) =
                            sel.iter().find(|(k, _)| *k == set)
                        {
                            variants.push(Src {
                                stack: s.stack,
                                path: format!("{}{{{set}={choice}}}", s.path),
                            });
                        }
                    }
                }
                for (field, out) in [("references", &mut refs), ("payload", &mut payloads)] {
                    if let Some(Value::ListOp(op)) = spec.field(field) {
                        for item in op.effective_items() {
                            let arc = match item {
                                Value::Reference(a) | Value::Payload(a) => *a,
                                _ => {
                                    return Err(UsdError::Malformed(format!(
                                        "{field} on {} holds a non-arc",
                                        s.path
                                    )))
                                }
                            };
                            out.push(self.resolve(&layer_id, s.stack, &arc)?);
                        }
                    }
                }
            }
        }
        let mut out: Vec<Src> = Vec::new();
        for s in sources.iter().cloned() {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        for group in [variants, refs, payloads] {
            if group.is_empty() {
                continue;
            }
            for s in self.gather(&group, depth + 1)? {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        Ok(out)
    }

    /// Turns an arc into a source: the target layer stack and prim path.
    fn resolve(&mut self, authoring_layer: &str, current_stack: usize, arc: &Arc) -> Result<Src> {
        if arc.layer_offset != (0.0, 1.0) {
            return Err(UsdError::Unsupported(format!(
                "non-identity layer offset on an arc to {}",
                arc.asset_path
            )));
        }
        let stack = if arc.asset_path.is_empty() {
            current_stack
        } else {
            let file = resolve_relative(authoring_layer, &arc.asset_path);
            let li = self.load(&file.display().to_string())?;
            self.stack_of(li)?
        };
        let path = if arc.prim_path.is_empty() {
            self.default_prim(stack)?
        } else {
            arc.prim_path.clone()
        };
        Ok(Src { stack, path })
    }

    fn default_prim(&self, stack: usize) -> Result<String> {
        let layer = self.stacks[stack][0];
        let name = self.layers[layer]
            .specs
            .iter()
            .find(|s| s.kind == SpecKind::PseudoRoot)
            .and_then(|s| s.field("defaultPrim"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                UsdError::Malformed(format!(
                    "{} has no defaultPrim for an arc",
                    self.layers[layer].identifier
                ))
            })?;
        Ok(format!("/{name}"))
    }

    fn compose_root(&mut self, stack: usize) -> Result<Stage> {
        let ids: Vec<String> = self.stacks[stack]
            .iter()
            .map(|&i| self.layers[i].identifier.clone())
            .collect();
        let root_src = Src {
            stack,
            path: "/".to_owned(),
        };
        let mut metadata: Vec<(String, Value)> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for at in self.specs_at(&root_src) {
            let spec = self.spec(at).clone();
            if spec.kind != SpecKind::PseudoRoot {
                continue;
            }
            if let Some(offsets) = spec.field("subLayerOffsets") {
                let identity = offsets.as_array().is_none_or(|a| {
                    a.iter().all(
                        |o| matches!(o, Value::LayerOffset(off, sc) if *off == 0.0 && *sc == 1.0),
                    )
                });
                if !identity {
                    return Err(UsdError::Unsupported("non-identity subLayerOffsets".into()));
                }
            }
            for (n, v) in &spec.fields {
                if !matches!(n.as_str(), "subLayers" | "subLayerOffsets" | "primChildren")
                    && !metadata.iter().any(|(m, _)| m == n)
                {
                    metadata.push((n.clone(), v.clone()));
                }
            }
            union_names(&mut names, spec.field("primChildren"));
        }
        let sources = self.gather(&[root_src], 0)?;
        for s in &sources {
            for at in self.specs_at(s) {
                let spec = self.spec(at).clone();
                union_names(&mut names, spec.field("primChildren"));
            }
        }
        let mut prims = Vec::new();
        for name in &names {
            let child: Vec<Src> = sources
                .iter()
                .map(|s| Src {
                    stack: s.stack,
                    path: child_path(&s.path, name),
                })
                .collect();
            match self.compose_prim(&child, &child_path("/", name))? {
                Some(p) => prims.push(p),
                None => {
                    return Err(UsdError::Malformed(format!(
                        "primChildren names undefined prim {}",
                        child_path("/", name)
                    )))
                }
            }
        }
        Ok(Stage {
            layers: ids,
            metadata: metadata.into_iter().collect(),
            prims,
        })
    }

    /// Composes the prim whose opinions are at `sources`, or `None` if no
    /// source authors it.
    fn compose_prim(&mut self, sources: &[Src], path: &str) -> Result<Option<Prim>> {
        let full = self.gather(sources, 0)?;
        let mut fields: Vec<(String, Value)> = Vec::new();
        let mut children: Vec<String> = Vec::new();
        let mut properties: Vec<String> = Vec::new();
        let mut any = false;
        for s in &full {
            for at in self.specs_at(s) {
                let spec = self.spec(at).clone();
                if !matches!(spec.kind, SpecKind::Prim | SpecKind::Variant) {
                    continue;
                }
                any = true;
                for (n, v) in &spec.fields {
                    if !fields.iter().any(|(m, _)| m == n) {
                        fields.push((n.clone(), v.clone()));
                    }
                }
                union_names(&mut children, spec.field("primChildren"));
                union_names(&mut properties, spec.field("properties"));
            }
        }
        if !any {
            return Ok(None);
        }

        let mut kids = Vec::new();
        for name in &children {
            let child: Vec<Src> = full
                .iter()
                .map(|s| Src {
                    stack: s.stack,
                    path: child_path(&s.path, name),
                })
                .collect();
            match self.compose_prim(&child, &child_path(path, name))? {
                Some(p) => kids.push(p),
                None => {
                    return Err(UsdError::Malformed(format!(
                        "primChildren names undefined prim {}",
                        child_path(path, name)
                    )))
                }
            }
        }

        let mut props = Vec::new();
        for name in &properties {
            props.push(self.compose_property(&full, path, name)?);
        }

        let mut specifier = None;
        let mut type_name = None;
        let mut kind = None;
        let mut api_schemas = Vec::new();
        let mut metadata = BTreeMap::new();
        for (n, v) in &fields {
            match (n.as_str(), v) {
                ("specifier", Value::Specifier(s)) => specifier = Some(*s),
                ("typeName", Value::Token(t)) => type_name = Some(t.clone()),
                ("kind", Value::Token(t)) => kind = Some(t.clone()),
                ("apiSchemas", Value::ListOp(op)) => {
                    api_schemas = effective_strings(op, path, "apiSchemas")?
                }
                (
                    "primChildren" | "properties" | "references" | "payload" | "variantSelection"
                    | "variantSetNames" | "variantChildren" | "variantSetChildren",
                    _,
                ) => {}
                _ => {
                    metadata.insert(n.clone(), v.clone());
                }
            }
        }
        Ok(Some(Prim {
            name: last_name(path),
            path: path.to_owned(),
            specifier,
            type_name,
            kind,
            api_schemas,
            properties: props,
            children: kids,
            metadata,
        }))
    }

    fn compose_property(&self, owners: &[Src], owner_path: &str, name: &str) -> Result<Property> {
        let path = format!("{owner_path}.{name}");
        let mut fields: Vec<(String, Value)> = Vec::new();
        let mut kind = PropertyKind::Attribute;
        for s in owners {
            let prop = Src {
                stack: s.stack,
                path: format!("{}.{name}", s.path),
            };
            for at in self.specs_at(&prop) {
                let spec = self.spec(at);
                if spec.kind == SpecKind::Relationship {
                    kind = PropertyKind::Relationship;
                }
                for (n, v) in &spec.fields {
                    if !fields.iter().any(|(m, _)| m == n) {
                        fields.push((n.clone(), v.clone()));
                    }
                }
            }
        }
        let mut out = Property {
            name: name.to_owned(),
            path: path.clone(),
            kind,
            type_name: None,
            variability: None,
            default: None,
            time_samples: None,
            connections: Vec::new(),
            targets: Vec::new(),
            metadata: BTreeMap::new(),
        };
        for (n, v) in &fields {
            match (n.as_str(), v) {
                ("typeName", Value::Token(t)) => out.type_name = Some(t.clone()),
                ("variability", Value::Variability(x)) => out.variability = Some(*x),
                ("default", _) => out.default = Some(v.clone()),
                ("timeSamples", Value::TimeSamples(ts)) => out.time_samples = Some(ts.clone()),
                ("connectionPaths", Value::ListOp(op)) => {
                    out.connections = effective_strings(op, &path, "connectionPaths")?
                }
                ("targetPaths", Value::ListOp(op)) => {
                    out.targets = effective_strings(op, &path, "targetPaths")?
                }
                _ => {
                    out.metadata.insert(n.clone(), v.clone());
                }
            }
        }
        Ok(out)
    }
}

/// Rejects spec kinds and arcs this composer does not implement, so that no
/// authored opinion is dropped without a report.
fn check_layer(layer: &Layer) -> Result<()> {
    for spec in &layer.specs {
        match spec.kind {
            SpecKind::Unknown
            | SpecKind::Connection
            | SpecKind::Expression
            | SpecKind::Mapper
            | SpecKind::MapperArg
            | SpecKind::RelationshipTarget => {
                return Err(UsdError::Unsupported(format!(
                    "{:?} spec {} in {}",
                    spec.kind, spec.path, layer.identifier
                )));
            }
            _ => {}
        }
        for arc in ["inherits", "specializes"] {
            if let Some(Value::ListOp(op)) = spec.field(arc) {
                if !op.effective_items().is_empty()
                    || !op.deleted.is_empty()
                    || !op.ordered.is_empty()
                {
                    return Err(UsdError::Unsupported(format!(
                        "{arc} arc on {} in {}",
                        spec.path, layer.identifier
                    )));
                }
            }
        }
    }
    Ok(())
}

fn sublayer_refs(layer: &Layer) -> Vec<String> {
    layer
        .specs
        .iter()
        .find(|s| s.kind == SpecKind::PseudoRoot)
        .and_then(|s| s.field("subLayers"))
        .and_then(Value::string_items)
        .unwrap_or_default()
}

fn union_names(into: &mut Vec<String>, v: Option<&Value>) {
    if let Some(items) = v.and_then(Value::string_items) {
        for it in items {
            if !into.contains(&it) {
                into.push(it);
            }
        }
    }
}

/// The effective item list of a list op. Deleted and ordered items are not
/// implemented and are reported rather than ignored.
fn effective_strings(op: &ListOp, at: &str, field: &str) -> Result<Vec<String>> {
    if !op.deleted.is_empty() || !op.ordered.is_empty() {
        return Err(UsdError::Unsupported(format!(
            "deleted or ordered items in {field} of {at}"
        )));
    }
    op.effective_items()
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                UsdError::Malformed(format!("{field} of {at} holds a non-text item"))
            })
        })
        .collect()
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn last_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or_default().to_owned()
}
