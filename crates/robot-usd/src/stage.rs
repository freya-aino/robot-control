//! Layer composition.
//!
//! The root layer's `subLayers` are loaded recursively, strongest first. A
//! prim is then composed from a list of *sources*: `(layer stack, spec path)`
//! pairs. The list starts with the prim's own specs and is extended, in USD's
//! LIVRPS order, by inherits, then variant selections, then references, then
//! payloads, then specializes. Each field takes the value of the strongest
//! source that authors it, except list-valued fields, which compose every
//! source's list op weakest first. Child and property orders are unions of
//! the sources' orders, strongest first.
//!
//! Implemented arcs: sublayers, inherits, variant selections, internal and
//! external references, payloads, and specializes, each arc with an optional
//! prim path (default prim when empty) and a layer offset applied to time
//! samples. Relationship targets and connections authored inside a referenced
//! or payloaded subtree are remapped into the composed namespace; targets
//! that point outside the arc's subtree are kept as authored. List edits
//! (explicit, add, prepend, append, delete, reorder) follow
//! `SdfListOp::ApplyOperations`. Standalone connection and relationship-target
//! specs are accepted when they author no fields of their own (the target
//! lists already carry their paths); ones with fields are reported as
//! [`UsdError::Unsupported`], as is every other construct outside this set.

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
    /// The `timeSamples` of an attribute, if authored, in stage time (layer
    /// offsets applied).
    pub time_samples: Option<Vec<(f64, Value)>>,
    /// Connection paths of an attribute (`connectionPaths`), remapped into
    /// the composed namespace.
    pub connections: Vec<String>,
    /// Target paths of a relationship (`targetPaths`), remapped into the
    /// composed namespace.
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

/// A layer time offset `(offset, scale)`: stage time = offset + scale * layer
/// time. Composition `outer ∘ inner` applies the inner mapping first.
type Offset = (f64, f64);

const IDENTITY: Offset = (0.0, 1.0);

fn compose_offset(outer: Offset, inner: Offset) -> Offset {
    (outer.0 + outer.1 * inner.0, outer.1 * inner.1)
}

/// A place a prim's opinions can come from: a spec path in a layer stack,
/// reached through arcs whose combined layer offset is `offset`.
#[derive(Debug, Clone, PartialEq)]
struct Src {
    stack: usize,
    path: String,
    offset: Offset,
    /// The namespace translation the arc behind this source implies:
    /// `(root in the source layer, root in the composed stage)`. `None`
    /// for a source that is not behind an arc (identity).
    root: Option<(String, String)>,
}

impl Src {
    /// Where this source's prim sits in the composed namespace.
    fn composed_path(&self) -> String {
        match &self.root {
            Some(roots) => remap_path(&self.path, roots),
            None => self.path.clone(),
        }
    }
}

#[derive(Default)]
struct Composer {
    layers: Vec<Layer>,
    /// Per layer: spec path to spec index.
    index: Vec<HashMap<String, usize>>,
    by_id: HashMap<String, usize>,
    /// Each stack: `(layer index, offset)`, strongest first.
    stacks: Vec<Vec<(usize, Offset)>>,
    stack_of_layer: HashMap<usize, usize>,
    in_progress: HashSet<usize>,
}

const MAX_NESTING: usize = 64;

/// The deepest prim nesting the recursive walks follow (the composed-tree
/// walk here, and the text parser's prim recursion): one stack frame per
/// level, so a bound keeps a crafted file from overflowing the stack.
pub(crate) const MAX_PRIM_DEPTH: usize = 256;

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

    /// The layer stack of a layer: the layer, then its sublayers' stacks,
    /// each member carrying the sublayer offsets composed down to it.
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
        let mut members: Vec<(usize, Offset)> = vec![(layer, IDENTITY)];
        let ident = self.layers[layer].identifier.clone();
        let subs = sublayer_refs(&self.layers[layer]);
        let offsets = sublayer_offsets(&self.layers[layer], subs.len())?;
        for (sub, off) in subs.into_iter().zip(offsets) {
            let file = resolve_relative(&ident, &sub).display().to_string();
            let li = self.load(&file)?;
            let st = self.stack_of(li)?;
            for (m, mo) in self.stacks[st].clone() {
                if !members.iter().any(|(e, _)| *e == m) {
                    members.push((m, compose_offset(off, mo)));
                }
            }
        }
        self.in_progress.remove(&layer);
        self.stacks.push(members);
        let id = self.stacks.len() - 1;
        self.stack_of_layer.insert(layer, id);
        Ok(id)
    }

    /// The specs at a source, strongest layer first, with the total offset
    /// from that layer's time to stage time.
    fn specs_at(&self, src: &Src) -> Vec<((usize, usize), Offset)> {
        self.stacks[src.stack]
            .iter()
            .filter_map(|&(li, off)| {
                self.index[li]
                    .get(&src.path)
                    .map(|&k| ((li, k), compose_offset(src.offset, off)))
            })
            .collect()
    }

    fn spec(&self, at: (usize, usize)) -> &Spec {
        &self.layers[at.0].specs[at.1]
    }

    /// The composed items of a list-op field across a source's specs
    /// (strongest first): every op applied weakest first.
    fn composed_list(&self, src: &Src, field: &str) -> Vec<Value> {
        let mut ops: Vec<ListOp> = Vec::new();
        for (at, _) in self.specs_at(src) {
            if let Some(Value::ListOp(op)) = self.spec(at).field(field) {
                ops.push((**op).clone());
            }
        }
        let mut items = Vec::new();
        for op in ops.iter().rev() {
            items = op.apply(&items);
        }
        items
    }

    /// Expands inherits, variant selections, references, payloads and
    /// specializes into an ordered source list, LIVRPS order.
    fn gather(
        &mut self,
        sources: &[Src],
        depth: usize,
        outer_selections: &[(String, String)],
    ) -> Result<Vec<Src>> {
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
        let mut inherits = Vec::new();
        let mut variants = Vec::new();
        let mut refs = Vec::new();
        let mut payloads = Vec::new();
        let mut specializes = Vec::new();
        // Variant selections compose strongest-wins per set name over the
        // *whole* source list: a stronger opinion (a local one, or a
        // stronger sublayer's) replaces a weaker one, it does not add to
        // it, so only the winning variant of each set becomes a source and
        // the deselected variant's opinions cannot leak into the prim.
        let mut chosen: Vec<(String, String)> = outer_selections.to_vec();
        for s in sources {
            for (at, _) in self.specs_at(s) {
                if let Some(Value::Dictionary(sel)) = self.spec(at).field("variantSelection") {
                    for (set, choice) in sel {
                        if let Value::String(choice) = choice {
                            if !chosen.iter().any(|(k, _)| k == set) {
                                chosen.push((set.clone(), choice.clone()));
                            }
                        }
                    }
                }
            }
        }
        for s in sources {
            for (field, out) in [
                ("inherits", &mut inherits),
                ("specializes", &mut specializes),
            ] {
                for item in self.composed_list(s, field) {
                    let path = item.as_str().map(str::to_owned).ok_or_else(|| {
                        UsdError::Malformed(format!("{field} on {} holds a non-path", s.path))
                    })?;
                    out.push(Src {
                        stack: s.stack,
                        root: Some((path.clone(), s.composed_path())),
                        path,
                        offset: s.offset,
                    });
                }
            }
            // This source's declared variant sets, each resolved to the
            // strongest selection over the whole source list.
            let mut sets: Vec<String> = Vec::new();
            for (at, _) in self.specs_at(s) {
                let spec = self.spec(at).clone();
                if let Some(Value::Dictionary(sel)) = spec.field("variantSelection") {
                    let mut names: Vec<String> = Vec::new();
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
                        if !sets.contains(&set) {
                            sets.push(set);
                        }
                    }
                }
            }
            for set in sets {
                if let Some((_, choice)) = chosen.iter().find(|(k, _)| *k == set) {
                    variants.push(Src {
                        stack: s.stack,
                        path: format!("{}{{{set}={choice}}}", s.path),
                        offset: s.offset,
                        root: Some((format!("{}{{{set}={choice}}}", s.path), s.composed_path())),
                    });
                }
            }
            for (field, out) in [("references", &mut refs), ("payload", &mut payloads)] {
                // The arcs of each authoring layer resolve against that
                // layer's own identifier, so resolve per spec, strongest
                // first, over the per-source composed list.
                let items = self.composed_list(s, field);
                for item in items {
                    let arc = match item {
                        Value::Reference(a) | Value::Payload(a) => *a,
                        _ => {
                            return Err(UsdError::Malformed(format!(
                                "{field} on {} holds a non-arc",
                                s.path
                            )))
                        }
                    };
                    let authoring = self.authoring_layer_of(s, field, &arc);
                    out.push(self.resolve(&authoring, s, &arc)?);
                }
            }
        }
        let mut out: Vec<Src> = Vec::new();
        for s in sources.iter().cloned() {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        for group in [inherits, variants, refs, payloads, specializes] {
            if group.is_empty() {
                continue;
            }
            for s in self.gather(&group, depth + 1, &chosen)? {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        Ok(out)
    }

    /// The identifier of the strongest layer in `src`'s stack whose list op
    /// for `field` mentions this arc; relative asset paths resolve against it.
    fn authoring_layer_of(&self, src: &Src, field: &str, arc: &Arc) -> String {
        let holds = |op: &ListOp| {
            [&op.explicit, &op.added, &op.prepended, &op.appended]
                .iter()
                .any(|items| {
                    items.iter().any(|v| match v {
                        Value::Reference(a) | Value::Payload(a) => **a == *arc,
                        _ => false,
                    })
                })
        };
        for (at, _) in self.specs_at(src) {
            if let Some(Value::ListOp(op)) = self.spec(at).field(field) {
                if holds(op) {
                    return self.layers[at.0].identifier.clone();
                }
            }
        }
        self.layers[self.stacks[src.stack][0].0].identifier.clone()
    }

    /// Turns an arc into a source: the target layer stack, prim path, and
    /// the arc's layer offset composed onto the source's.
    fn resolve(&mut self, authoring_layer: &str, from: &Src, arc: &Arc) -> Result<Src> {
        let stack = if arc.asset_path.is_empty() {
            from.stack
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
        Ok(Src {
            stack,
            root: Some((path.clone(), from.composed_path())),
            path,
            offset: compose_offset(from.offset, arc.layer_offset),
        })
    }

    fn default_prim(&self, stack: usize) -> Result<String> {
        let layer = self.stacks[stack][0].0;
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
            .map(|&(i, _)| self.layers[i].identifier.clone())
            .collect();
        let root_src = Src {
            stack,
            path: "/".to_owned(),
            offset: IDENTITY,
            root: None,
        };
        let mut metadata: Vec<(String, Value)> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for (at, _) in self.specs_at(&root_src) {
            let spec = self.spec(at).clone();
            if spec.kind != SpecKind::PseudoRoot {
                continue;
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
        let sources = self.gather(&[root_src], 0, &[])?;
        for s in &sources {
            for (at, _) in self.specs_at(s) {
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
                    offset: s.offset,
                    root: s.root.clone(),
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
        // The walk recurses once per prim-tree level; the bound turns a
        // crafted file's hundred-thousand-deep chain into an error instead
        // of a stack overflow.
        if path.split('/').count() > MAX_PRIM_DEPTH {
            return Err(UsdError::Unsupported(format!(
                "prims nested deeper than {MAX_PRIM_DEPTH} at {path}"
            )));
        }
        let full = self.gather(sources, 0, &[])?;
        let mut fields: Vec<(String, Value)> = Vec::new();
        let mut children: Vec<String> = Vec::new();
        let mut properties: Vec<String> = Vec::new();
        let mut api_ops: Vec<ListOp> = Vec::new();
        let mut any = false;
        for s in &full {
            for (at, _) in self.specs_at(s) {
                let spec = self.spec(at).clone();
                if !matches!(spec.kind, SpecKind::Prim | SpecKind::Variant) {
                    continue;
                }
                any = true;
                for (n, v) in &spec.fields {
                    if n == "apiSchemas" {
                        if let Value::ListOp(op) = v {
                            api_ops.push((**op).clone());
                        }
                        continue;
                    }
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
                    offset: s.offset,
                    root: s.root.clone(),
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

        let mut api_schemas_items = Vec::new();
        for op in api_ops.iter().rev() {
            api_schemas_items = op.apply(&api_schemas_items);
        }
        let api_schemas = strings_of(&api_schemas_items, path, "apiSchemas")?;

        let mut specifier = None;
        let mut type_name = None;
        let mut kind = None;
        let mut metadata = BTreeMap::new();
        for (n, v) in &fields {
            match (n.as_str(), v) {
                ("specifier", Value::Specifier(s)) => specifier = Some(*s),
                ("typeName", Value::Token(t)) => type_name = Some(t.clone()),
                ("kind", Value::Token(t)) => kind = Some(t.clone()),
                (
                    "primChildren" | "properties" | "references" | "payload" | "inherits"
                    | "specializes" | "variantSelection" | "variantSetNames" | "variantChildren"
                    | "variantSetChildren",
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
        let mut fields: Vec<(String, Value, Offset)> = Vec::new();
        let mut connection_ops: Vec<(ListOp, (String, String))> = Vec::new();
        let mut target_ops: Vec<(ListOp, (String, String))> = Vec::new();
        let mut kind = PropertyKind::Attribute;
        for s in owners {
            // The namespace translation recorded when the arc was resolved;
            // a local source translates as the identity.
            let roots = s
                .root
                .clone()
                .unwrap_or_else(|| (owner_path.to_owned(), owner_path.to_owned()));
            let prop = Src {
                stack: s.stack,
                path: format!("{}.{name}", s.path),
                offset: s.offset,
                root: s.root.clone(),
            };
            for (at, off) in self.specs_at(&prop) {
                let spec = self.spec(at);
                if spec.kind == SpecKind::Relationship {
                    kind = PropertyKind::Relationship;
                }
                for (n, v) in &spec.fields {
                    match (n.as_str(), v) {
                        ("connectionPaths", Value::ListOp(op)) => {
                            connection_ops.push(((**op).clone(), roots.clone()));
                        }
                        ("targetPaths", Value::ListOp(op)) => {
                            target_ops.push(((**op).clone(), roots.clone()));
                        }
                        _ => {
                            if !fields.iter().any(|(m, _, _)| m == n) {
                                fields.push((n.clone(), v.clone(), off));
                            }
                        }
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
            connections: compose_paths(&connection_ops, &path, "connectionPaths")?,
            targets: compose_paths(&target_ops, &path, "targetPaths")?,
            metadata: BTreeMap::new(),
        };
        for (n, v, off) in &fields {
            match (n.as_str(), v) {
                ("typeName", Value::Token(t)) => out.type_name = Some(t.clone()),
                ("variability", Value::Variability(x)) => out.variability = Some(*x),
                ("default", _) => out.default = Some(v.clone()),
                ("timeSamples", Value::TimeSamples(ts)) => {
                    out.time_samples = Some(
                        ts.iter()
                            .map(|(t, v)| (off.0 + off.1 * t, v.clone()))
                            .collect(),
                    );
                }
                _ => {
                    out.metadata.insert(n.clone(), v.clone());
                }
            }
        }
        Ok(out)
    }
}

/// Remaps a path authored in a source namespace into the composed namespace.
/// Paths outside the arc's subtree are kept as authored.
fn remap_path(authored: &str, (from, to): &(String, String)) -> String {
    if from == to || from.is_empty() {
        return authored.to_owned();
    }
    let (prim, prop) = match authored.find('.') {
        Some(i) => (&authored[..i], &authored[i..]),
        None => (authored, ""),
    };
    if prim == from {
        format!("{to}{prop}")
    } else if let Some(rest) = prim.strip_prefix(&format!("{from}/")) {
        format!("{to}/{rest}{prop}")
    } else {
        authored.to_owned()
    }
}

/// Composes path-list ops gathered strongest first, remapping each op's items
/// into the composed namespace before applying it.
fn compose_paths(ops: &[(ListOp, (String, String))], at: &str, field: &str) -> Result<Vec<String>> {
    let remap_items = |items: &[Value], roots: &(String, String)| -> Result<Vec<Value>> {
        items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|p| Value::Path(remap_path(p, roots)))
                    .ok_or_else(|| {
                        UsdError::Malformed(format!("{field} of {at} holds a non-text item"))
                    })
            })
            .collect()
    };
    let mut items: Vec<Value> = Vec::new();
    for (op, roots) in ops.iter().rev() {
        let remapped = ListOp {
            is_explicit: op.is_explicit,
            explicit: remap_items(&op.explicit, roots)?,
            added: remap_items(&op.added, roots)?,
            prepended: remap_items(&op.prepended, roots)?,
            appended: remap_items(&op.appended, roots)?,
            deleted: remap_items(&op.deleted, roots)?,
            ordered: remap_items(&op.ordered, roots)?,
        };
        items = remapped.apply(&items);
    }
    strings_of(&items, at, field)
}

fn strings_of(items: &[Value], at: &str, field: &str) -> Result<Vec<String>> {
    items
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                UsdError::Malformed(format!("{field} of {at} holds a non-text item"))
            })
        })
        .collect()
}

/// Rejects spec kinds this composer does not implement, so that no authored
/// opinion is dropped without a report. Standalone connection and
/// relationship-target specs are accepted when they carry no fields of their
/// own: their paths are already in the owning property's list.
fn check_layer(layer: &Layer) -> Result<()> {
    for spec in &layer.specs {
        match spec.kind {
            SpecKind::Unknown | SpecKind::Expression | SpecKind::Mapper | SpecKind::MapperArg => {
                return Err(UsdError::Unsupported(format!(
                    "{:?} spec {} in {}",
                    spec.kind, spec.path, layer.identifier
                )));
            }
            SpecKind::Connection | SpecKind::RelationshipTarget if !spec.fields.is_empty() => {
                return Err(UsdError::Unsupported(format!(
                    "{:?} spec {} in {} authors fields of its own",
                    spec.kind, spec.path, layer.identifier
                )));
            }
            _ => {}
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

/// The per-sublayer offsets, identity when unauthored. A `subLayerOffsets`
/// whose length disagrees with `subLayers` is malformed.
fn sublayer_offsets(layer: &Layer, count: usize) -> Result<Vec<Offset>> {
    let Some(spec) = layer.specs.iter().find(|s| s.kind == SpecKind::PseudoRoot) else {
        return Ok(vec![IDENTITY; count]);
    };
    match spec.field("subLayerOffsets").and_then(Value::as_array) {
        None => Ok(vec![IDENTITY; count]),
        Some(a) => {
            if a.len() != count {
                return Err(UsdError::Malformed(format!(
                    "{}: {} subLayerOffsets for {} subLayers",
                    layer.identifier,
                    a.len(),
                    count
                )));
            }
            a.iter()
                .map(|o| match o {
                    Value::LayerOffset(off, sc) => Ok((*off, *sc)),
                    _ => Err(UsdError::Malformed(format!(
                        "{}: subLayerOffsets holds a non-offset",
                        layer.identifier
                    ))),
                })
                .collect()
        }
    }
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
