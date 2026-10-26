//! A minimal XML subset parser and a URDF subset importer.
//!
//! SPECS.md Section 7.1: "URDF import is an initial interchange target, not a
//! complete dynamics specification. Store missing actuator/contact metadata
//! separately. Unsupported model features must be rejected explicitly rather
//! than silently dropped." Appendix B.2.4 shortlists `urdf-rs`, and Appendix
//! B.6 admits no candidate until its gates pass, so this module is the
//! authored interim path: a small XML reader plus the URDF-specific semantic
//! layer this project owns regardless of which parser is eventually qualified
//! (Section 7.1's closing sentence: "The platform owns semantic validation").
//!
//! Supported: `<robot>`, `<link>` with `<inertial>`/`<mass>`/`<inertia>` and
//! `<collision><geometry>` (`box`/`sphere`/`cylinder`/`mesh`), `<joint
//! type="revolute|continuous|prismatic|fixed">` with `<parent>`, `<child>`,
//! `<origin>`, `<axis>` and `<limit>`.
//!
//! Explicitly rejected by name, never silently dropped: `floating` and
//! `planar` joints, a `mimic` element, and a `<transmission>` block — all
//! require modeling this importer does not attempt, and Section 7.1 asks for
//! that gap to be visible rather than guessed at.

use crate::actuator::{ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission};
use crate::joint::{JointKind, JointLimits, JointSpec};
use crate::link::{CollisionGeometry, CollisionShape, InertiaTensor, LinkSpec, MassProperties};
use crate::pose::RigidPose;
use core::fmt;
use core::ops::Mul;
use robot_types::frames::{Quat, Vec3};
use robot_types::ids::{ActuatorId, JointId, LinkId, Name, NameError};
use robot_types::units::{Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range};

/// Why URDF import failed.
#[derive(Clone, PartialEq, Debug)]
pub enum UrdfError {
    /// The XML did not parse.
    Xml(XmlError),
    /// A required attribute or child element was absent.
    Missing {
        /// The element.
        element: &'static str,
        /// The missing part.
        what: &'static str,
    },
    /// An attribute value did not parse as expected.
    BadValue {
        /// The element.
        element: &'static str,
        /// The attribute.
        attribute: &'static str,
        /// The text found.
        text: String,
    },
    /// A construct outside the supported URDF subset was present.
    ///
    /// Named explicitly rather than skipped, per SPECS.md Section 7.1.
    Unsupported {
        /// The construct's name, e.g. `"floating joint"` or `"mimic"`.
        construct: String,
        /// The entity it appeared on, where applicable.
        context: String,
    },
    /// Two links or joints shared a name.
    DuplicateName {
        /// The category.
        category: &'static str,
        /// The repeated name.
        name: String,
    },
    /// A joint names a link that is not defined.
    UnknownLink {
        /// The joint.
        joint: String,
        /// Which side.
        side: &'static str,
        /// The missing link name.
        link: String,
    },
    /// No link is the root (every link has a parent joint, or there are
    /// several unconnected roots).
    AmbiguousOrMissingRoot {
        /// Candidate roots found.
        candidates: Vec<String>,
    },
    /// A name failed the project's name validation.
    Name(NameError),
}

impl fmt::Display for UrdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(e) => write!(f, "{e}"),
            Self::Missing { element, what } => write!(f, "<{element}> is missing {what}"),
            Self::BadValue {
                element,
                attribute,
                text,
            } => {
                write!(f, "<{element} {attribute}={text:?}> did not parse")
            }
            Self::Unsupported { construct, context } => write!(
                f,
                "{construct} on {context} is outside the supported URDF subset"
            ),
            Self::DuplicateName { category, name } => {
                write!(f, "duplicate {category} name {name:?}")
            }
            Self::UnknownLink { joint, side, link } => {
                write!(f, "joint {joint:?} names unknown {side} link {link:?}")
            }
            Self::AmbiguousOrMissingRoot { candidates } => write!(
                f,
                "expected exactly one root link, found {}: {candidates:?}",
                candidates.len()
            ),
            Self::Name(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for UrdfError {}

impl From<XmlError> for UrdfError {
    fn from(e: XmlError) -> Self {
        Self::Xml(e)
    }
}
impl From<NameError> for UrdfError {
    fn from(e: NameError) -> Self {
        Self::Name(e)
    }
}

// ---------------------------------------------------------------------
// Minimal XML subset
// ---------------------------------------------------------------------

/// A parsed XML element.
#[derive(Clone, Debug)]
pub struct XmlElement {
    /// Tag name.
    pub tag: String,
    /// Attributes, in document order.
    pub attrs: Vec<(String, String)>,
    /// Child elements, in document order. Text content is discarded: the
    /// URDF subset this module reads carries its data in attributes.
    pub children: Vec<XmlElement>,
}

impl XmlElement {
    /// The first child with the given tag.
    #[must_use]
    pub fn child(&self, tag: &str) -> Option<&XmlElement> {
        self.children.iter().find(|c| c.tag == tag)
    }

    /// All children with the given tag.
    pub fn children_named<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a XmlElement> {
        self.children.iter().filter(move |c| c.tag == tag)
    }

    /// An attribute's value.
    #[must_use]
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Why XML parsing failed.
#[derive(Clone, PartialEq, Debug)]
pub enum XmlError {
    /// A syntax error at a byte offset.
    Syntax {
        /// What was expected.
        expected: &'static str,
        /// Byte offset.
        offset: usize,
    },
    /// A closing tag did not match its opening tag.
    MismatchedTag {
        /// The opening tag.
        opened: String,
        /// The closing tag found.
        closed: String,
    },
    /// The document had no root element, or more than one.
    NotSingleRoot,
    /// A construct this reader does not support, such as CDATA or a
    /// processing instruction.
    Unsupported {
        /// The construct.
        construct: &'static str,
        /// Byte offset.
        offset: usize,
    },
}

impl fmt::Display for XmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { expected, offset } => {
                write!(f, "expected {expected} at byte {offset}")
            }
            Self::MismatchedTag { opened, closed } => {
                write!(f, "<{opened}> closed by </{closed}>")
            }
            Self::NotSingleRoot => write!(f, "document does not have exactly one root element"),
            Self::Unsupported { construct, offset } => {
                write!(f, "{construct} at byte {offset} is not supported")
            }
        }
    }
}

impl core::error::Error for XmlError {}

/// Parses a small XML subset: elements, attributes, self-closing tags,
/// comments and a `<?xml ... ?>` prologue. No namespaces, no CDATA, no
/// entities beyond the five predefined ones, no text content retained.
pub fn parse_xml(input: &str) -> Result<XmlElement, XmlError> {
    let mut parser = XmlParser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    parser.skip_trivia();
    if parser.bytes[parser.pos..].starts_with(b"<?") {
        let end = parser.bytes[parser.pos..]
            .windows(2)
            .position(|w| w == b"?>")
            .ok_or(XmlError::Syntax {
                expected: "?>",
                offset: parser.pos,
            })?;
        parser.pos += end + 2;
        parser.skip_trivia();
    }
    let root = parser.element()?;
    parser.skip_trivia();
    if parser.pos != parser.bytes.len() {
        return Err(XmlError::NotSingleRoot);
    }
    Ok(root)
}

struct XmlParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl XmlParser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_trivia(&mut self) {
        loop {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
                self.pos += 1;
            }
            if self.bytes[self.pos..].starts_with(b"<!--") {
                let end = self.bytes[self.pos..]
                    .windows(3)
                    .position(|w| w == b"-->")
                    .map(|p| self.pos + p + 3);
                match end {
                    Some(e) => self.pos = e,
                    None => self.pos = self.bytes.len(),
                }
            } else {
                break;
            }
        }
    }

    fn name(&mut self) -> Result<String, XmlError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b':' || c == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(XmlError::Syntax {
                expected: "element or attribute name",
                offset: start,
            });
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn element(&mut self) -> Result<XmlElement, XmlError> {
        if self.peek() != Some(b'<') {
            return Err(XmlError::Syntax {
                expected: "'<'",
                offset: self.pos,
            });
        }
        self.pos += 1;
        let tag = self.name()?;
        let mut attrs = Vec::new();
        loop {
            self.skip_trivia();
            match self.peek() {
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() != Some(b'>') {
                        return Err(XmlError::Syntax {
                            expected: "'>'",
                            offset: self.pos,
                        });
                    }
                    self.pos += 1;
                    return Ok(XmlElement {
                        tag,
                        attrs,
                        children: Vec::new(),
                    });
                }
                Some(b'>') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    let attr_name = self.name()?;
                    self.skip_trivia();
                    if self.peek() != Some(b'=') {
                        return Err(XmlError::Syntax {
                            expected: "'='",
                            offset: self.pos,
                        });
                    }
                    self.pos += 1;
                    self.skip_trivia();
                    let value = self.quoted_value()?;
                    attrs.push((attr_name, value));
                }
                None => {
                    return Err(XmlError::Syntax {
                        expected: "'>' or attribute",
                        offset: self.pos,
                    })
                }
            }
        }

        let mut children = Vec::new();
        loop {
            self.skip_trivia();
            match self.peek() {
                Some(b'<') if self.bytes[self.pos..].starts_with(b"</") => {
                    self.pos += 2;
                    let close_tag = self.name()?;
                    self.skip_trivia();
                    if self.peek() != Some(b'>') {
                        return Err(XmlError::Syntax {
                            expected: "'>'",
                            offset: self.pos,
                        });
                    }
                    self.pos += 1;
                    if close_tag != tag {
                        return Err(XmlError::MismatchedTag {
                            opened: tag,
                            closed: close_tag,
                        });
                    }
                    return Ok(XmlElement {
                        tag,
                        attrs,
                        children,
                    });
                }
                Some(b'<') if self.bytes[self.pos..].starts_with(b"<![CDATA[") => {
                    return Err(XmlError::Unsupported {
                        construct: "CDATA",
                        offset: self.pos,
                    })
                }
                Some(b'<') => children.push(self.element()?),
                Some(_) => {
                    // Text content: skip to the next '<' without retaining it.
                    while let Some(c) = self.peek() {
                        if c == b'<' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                None => {
                    return Err(XmlError::Syntax {
                        expected: "closing tag",
                        offset: self.pos,
                    })
                }
            }
        }
    }

    fn quoted_value(&mut self) -> Result<String, XmlError> {
        let quote = self
            .peek()
            .filter(|c| *c == b'"' || *c == b'\'')
            .ok_or(XmlError::Syntax {
                expected: "quoted attribute value",
                offset: self.pos,
            })?;
        self.pos += 1;
        let start = self.pos;
        while self.peek().is_some_and(|c| c != quote) {
            self.pos += 1;
        }
        if self.peek() != Some(quote) {
            return Err(XmlError::Syntax {
                expected: "closing quote",
                offset: self.pos,
            });
        }
        let raw =
            core::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| XmlError::Syntax {
                expected: "valid UTF-8",
                offset: start,
            })?;
        self.pos += 1;
        Ok(unescape_entities(raw))
    }
}

fn unescape_entities(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

// ---------------------------------------------------------------------
// URDF semantics
// ---------------------------------------------------------------------

/// The result of a URDF import: the kinematic/dynamic facts URDF carries,
/// which the caller combines with the actuator and sensor facts URDF does
/// not (SPECS.md Section 7.1: "Store missing actuator/contact metadata
/// separately").
#[derive(Clone, Debug)]
pub struct ImportedKinematics {
    /// Imported links.
    pub links: Vec<LinkSpec>,
    /// Imported joints.
    pub joints: Vec<JointSpec>,
    /// The link with no parent joint.
    pub root_link: LinkId,
}

/// Parses three whitespace-separated floats, as URDF's `xyz`/`rpy` attributes
/// use.
fn parse_vec3(
    element: &'static str,
    attribute: &'static str,
    text: &str,
) -> Result<Vec3, UrdfError> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(UrdfError::BadValue {
            element,
            attribute,
            text: text.to_owned(),
        });
    }
    let mut v = [0.0; 3];
    for (slot, part) in v.iter_mut().zip(&parts) {
        *slot = part.parse::<f64>().map_err(|_| UrdfError::BadValue {
            element,
            attribute,
            text: text.to_owned(),
        })?;
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(UrdfError::BadValue {
            element,
            attribute,
            text: text.to_owned(),
        });
    }
    Ok(Vec3::from_xyz(v))
}

fn parse_f64(element: &'static str, attribute: &'static str, text: &str) -> Result<f64, UrdfError> {
    text.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| UrdfError::BadValue {
            element,
            attribute,
            text: text.to_owned(),
        })
}

/// Parses an `<origin xyz=".." rpy=".."/>` element into a [`RigidPose`].
///
/// URDF's `rpy` is extrinsic roll-pitch-yaw about fixed X, Y, Z axes applied
/// in that order, which composes as `Rz * Ry * Rx` on a column vector; this
/// is the one rotation convention this importer assumes, because it is
/// URDF's own fixed convention rather than a project-wide one (C-04 remains
/// about canonical *project* frames, not about what URDF itself defines).
fn parse_origin(origin: Option<&XmlElement>) -> Result<RigidPose, UrdfError> {
    let Some(origin) = origin else {
        return Ok(RigidPose::IDENTITY);
    };
    let xyz = match origin.attr("xyz") {
        Some(t) => parse_vec3("origin", "xyz", t)?,
        None => Vec3::ZERO,
    };
    let rpy = match origin.attr("rpy") {
        Some(t) => parse_vec3("origin", "rpy", t)?,
        None => Vec3::ZERO,
    };
    let rotation = Quat::from_rz(Radians(rpy.z))
        .unwrap_or(Quat::IDENTITY)
        .mul(Quat::from_ry(Radians(rpy.y)).unwrap_or(Quat::IDENTITY))
        .mul(Quat::from_rx(Radians(rpy.x)).unwrap_or(Quat::IDENTITY));
    Ok(RigidPose::new(xyz, rotation))
}

fn parse_link(element: &XmlElement) -> Result<LinkSpec, UrdfError> {
    let name_text = element.attr("name").ok_or(UrdfError::Missing {
        element: "link",
        what: "name attribute",
    })?;
    let name = Name::new(name_text)?;

    let mass = match element.child("inertial") {
        None => MassProperties::MASSLESS,
        Some(inertial) => {
            let mass_val = inertial
                .child("mass")
                .and_then(|m| m.attr("value"))
                .map(|v| parse_f64("mass", "value", v))
                .transpose()?
                .unwrap_or(0.0);
            let com = parse_origin(inertial.child("origin"))?.translation;
            let i = inertial.child("inertia");
            let get = |attr: &'static str| -> Result<f64, UrdfError> {
                match i.and_then(|e| e.attr(attr)) {
                    Some(v) => parse_f64("inertia", attr, v),
                    None => Ok(0.0),
                }
            };
            MassProperties {
                mass: Kilograms(mass_val),
                center_of_mass: com,
                inertia: InertiaTensor {
                    ixx: get("ixx")?,
                    iyy: get("iyy")?,
                    izz: get("izz")?,
                    ixy: get("ixy")?,
                    ixz: get("ixz")?,
                    iyz: get("iyz")?,
                },
            }
        }
    };

    let mut collision = Vec::new();
    for shape_el in element.children_named("collision") {
        let pose = parse_origin(shape_el.child("origin"))?;
        let geometry_el = shape_el.child("geometry").ok_or(UrdfError::Missing {
            element: "collision",
            what: "geometry",
        })?;
        let geometry = if let Some(b) = geometry_el.child("box") {
            let size = parse_vec3(
                "box",
                "size",
                b.attr("size").ok_or(UrdfError::Missing {
                    element: "box",
                    what: "size attribute",
                })?,
            )?;
            CollisionGeometry::Box {
                half_extents: size * 0.5,
            }
        } else if let Some(s) = geometry_el.child("sphere") {
            let radius = parse_f64(
                "sphere",
                "radius",
                s.attr("radius").ok_or(UrdfError::Missing {
                    element: "sphere",
                    what: "radius attribute",
                })?,
            )?;
            CollisionGeometry::Sphere { radius }
        } else if let Some(c) = geometry_el.child("cylinder") {
            let radius = parse_f64(
                "cylinder",
                "radius",
                c.attr("radius").ok_or(UrdfError::Missing {
                    element: "cylinder",
                    what: "radius attribute",
                })?,
            )?;
            let length = parse_f64(
                "cylinder",
                "length",
                c.attr("length").ok_or(UrdfError::Missing {
                    element: "cylinder",
                    what: "length attribute",
                })?,
            )?;
            CollisionGeometry::Cylinder { radius, length }
        } else if let Some(m) = geometry_el.child("mesh") {
            let filename = m.attr("filename").ok_or(UrdfError::Missing {
                element: "mesh",
                what: "filename attribute",
            })?;
            let scale = match m.attr("scale") {
                Some(t) => parse_vec3("mesh", "scale", t)?,
                None => Vec3::new(1.0, 1.0, 1.0),
            };
            // The filename, often a `package://` URI, is retained as the
            // asset reference text. Resolving it to a content-addressed blob
            // is `robot-artifacts`' job, not this importer's.
            CollisionGeometry::Mesh {
                content_id: filename.to_owned(),
                scale,
            }
        } else {
            return Err(UrdfError::Unsupported {
                construct: "geometry shape".to_owned(),
                context: format!("link {name_text:?}"),
            });
        };
        collision.push(CollisionShape { geometry, pose });
    }

    Ok(LinkSpec {
        id: LinkId::new(0),
        name,
        mass,
        collision,
    })
}

fn parse_joint(element: &XmlElement) -> Result<(JointSpec, String, String), UrdfError> {
    let name_text = element.attr("name").ok_or(UrdfError::Missing {
        element: "joint",
        what: "name attribute",
    })?;
    let name = Name::new(name_text)?;
    let type_text = element.attr("type").ok_or(UrdfError::Missing {
        element: "joint",
        what: "type attribute",
    })?;

    let kind = match type_text {
        "revolute" => JointKind::Revolute,
        "continuous" => JointKind::Continuous,
        "prismatic" => JointKind::Prismatic,
        "fixed" => JointKind::Fixed,
        "floating" | "planar" => {
            return Err(UrdfError::Unsupported {
                construct: format!("{type_text} joint"),
                context: format!("joint {name_text:?}"),
            })
        }
        other => {
            return Err(UrdfError::Unsupported {
                construct: format!("joint type {other:?}"),
                context: format!("joint {name_text:?}"),
            })
        }
    };

    if element.child("mimic").is_some() {
        return Err(UrdfError::Unsupported {
            construct: "mimic".to_owned(),
            context: format!("joint {name_text:?}"),
        });
    }

    let parent = element
        .child("parent")
        .and_then(|p| p.attr("link"))
        .ok_or(UrdfError::Missing {
            element: "joint",
            what: "<parent link=\"..\"/>",
        })?
        .to_owned();
    let child = element
        .child("child")
        .and_then(|c| c.attr("link"))
        .ok_or(UrdfError::Missing {
            element: "joint",
            what: "<child link=\"..\"/>",
        })?
        .to_owned();

    let origin = parse_origin(element.child("origin"))?;
    let axis = match element.child("axis").and_then(|a| a.attr("xyz")) {
        Some(t) => parse_vec3("axis", "xyz", t)?,
        None => Vec3::unit_z(),
    };

    let limit_el = element.child("limit");
    let velocity = limit_el
        .and_then(|l| l.attr("velocity"))
        .map(|v| parse_f64("limit", "velocity", v))
        .transpose()?
        .unwrap_or(0.0);
    let effort = limit_el
        .and_then(|l| l.attr("effort"))
        .map(|v| parse_f64("limit", "effort", v))
        .transpose()?
        .unwrap_or(0.0);

    let limits = if kind.is_bounded() {
        let lower = limit_el
            .and_then(|l| l.attr("lower"))
            .map(|v| parse_f64("limit", "lower", v))
            .transpose()?
            .ok_or(UrdfError::Missing {
                element: "limit",
                what: "lower attribute",
            })?;
        let upper = limit_el
            .and_then(|l| l.attr("upper"))
            .map(|v| parse_f64("limit", "upper", v))
            .transpose()?
            .ok_or(UrdfError::Missing {
                element: "limit",
                what: "upper attribute",
            })?;
        let range = Range::<Radians>::new(Radians(lower), Radians(upper)).map_err(|_| {
            UrdfError::BadValue {
                element: "limit",
                attribute: "lower/upper",
                text: format!("{lower} {upper}"),
            }
        })?;
        JointLimits::bounded(range, RadiansPerSecond(velocity), NewtonMetres(effort))
    } else {
        JointLimits::unbounded(RadiansPerSecond(velocity), NewtonMetres(effort))
    };

    let spec = JointSpec {
        id: JointId::new(0),
        name,
        kind,
        parent: LinkId::new(0),
        child: LinkId::new(0),
        origin,
        axis,
        limits,
        sign: 1.0,
        zero_offset: 0.0,
    };
    Ok((spec, parent, child))
}

/// Imports the kinematic and mass structure from a URDF document's root
/// `<robot>` element.
///
/// This assigns fresh [`LinkId`]/[`JointId`] values in document order; a
/// caller that needs IDs stable across re-imports supplies its own mapping
/// afterward, since URDF itself has no stable numeric identity concept
/// (SPECS.md Section 7.1's typed, stable IDs are this project's addition).
pub fn import_kinematics(xml: &XmlElement) -> Result<ImportedKinematics, UrdfError> {
    if xml.tag != "robot" {
        return Err(UrdfError::Missing {
            element: "robot",
            what: "as the document root",
        });
    }
    if xml.child("transmission").is_some() {
        return Err(UrdfError::Unsupported {
            construct: "transmission".to_owned(),
            context: "robot".to_owned(),
        });
    }

    let mut links = Vec::new();
    let mut link_names: Vec<String> = Vec::new();
    for (index, link_el) in xml.children_named("link").enumerate() {
        let mut spec = parse_link(link_el)?;
        if link_names.contains(&spec.name.to_string()) {
            return Err(UrdfError::DuplicateName {
                category: "link",
                name: spec.name.to_string(),
            });
        }
        spec.id = LinkId::new(index as u32);
        link_names.push(spec.name.to_string());
        links.push(spec);
    }

    let mut joints = Vec::new();
    let mut joint_names: Vec<String> = Vec::new();
    for (index, joint_el) in xml.children_named("joint").enumerate() {
        let (mut spec, parent_name, child_name) = parse_joint(joint_el)?;
        if joint_names.contains(&spec.name.to_string()) {
            return Err(UrdfError::DuplicateName {
                category: "joint",
                name: spec.name.to_string(),
            });
        }
        let parent_id = links
            .iter()
            .find(|l| l.name.as_str() == parent_name)
            .map(|l| l.id)
            .ok_or_else(|| UrdfError::UnknownLink {
                joint: spec.name.to_string(),
                side: "parent",
                link: parent_name.clone(),
            })?;
        let child_id = links
            .iter()
            .find(|l| l.name.as_str() == child_name)
            .map(|l| l.id)
            .ok_or_else(|| UrdfError::UnknownLink {
                joint: spec.name.to_string(),
                side: "child",
                link: child_name.clone(),
            })?;
        spec.id = JointId::new(index as u32);
        spec.parent = parent_id;
        spec.child = child_id;
        joint_names.push(spec.name.to_string());
        joints.push(spec);
    }

    let roots: Vec<String> = links
        .iter()
        .filter(|l| !joints.iter().any(|j| j.child == l.id))
        .map(|l| l.name.to_string())
        .collect();
    if roots.len() != 1 {
        return Err(UrdfError::AmbiguousOrMissingRoot { candidates: roots });
    }
    let root_link = links
        .iter()
        .find(|l| l.name.as_str() == roots[0])
        .expect("found above")
        .id;

    Ok(ImportedKinematics {
        links,
        joints,
        root_link,
    })
}

/// Imports from a URDF document's text in one call.
pub fn import_from_str(text: &str) -> Result<ImportedKinematics, UrdfError> {
    let xml = parse_xml(text)?;
    import_kinematics(&xml)
}

/// Builds a minimal actuator specification for an imported joint, using the
/// joint's own declared velocity/effort limit as the actuator's envelope.
///
/// This is a convenience for fixtures and early integration only. SPECS.md
/// Section 7.1 is explicit that URDF does not carry actuator metadata such as
/// transmission ratio, gains, thermal limits or command latency — a real
/// profile supplies [`ActuatorSpec`] values authored against the vendor's own
/// documentation, not derived from a joint limit.
#[must_use]
pub fn placeholder_actuator(index: u32, joint: &JointSpec) -> ActuatorSpec {
    ActuatorSpec {
        id: ActuatorId::new(index),
        name: joint.name.clone(),
        joint: joint.id,
        transmission: Transmission::DIRECT,
        envelope: TorqueSpeedEnvelope {
            stall_torque: joint.limits.effort,
            corner_speed: joint.limits.velocity * 0.5,
            no_load_speed: joint.limits.velocity.max(RadiansPerSecond(1e-6)),
        },
        gains: Some(PdGains { kp: 0.0, kd: 0.0 }),
        command_latency: robot_types::time::Duration::ZERO,
        thermal: ThermalLimits {
            limit: robot_types::units::Celsius(1000.0),
            ambient: robot_types::units::Celsius(25.0),
            time_constant: robot_types::time::Duration::from_secs(1),
            current_at_limit: robot_types::units::Amperes(1.0),
        },
        current_uncertainty: robot_types::units::Amperes(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_ARM: &str = r#"
<?xml version="1.0"?>
<robot name="fixture_arm">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
    </collision>
  </link>
  <link name="upper">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="lower">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.3"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="5.0" effort="20.0"/>
  </joint>
  <joint name="elbow" type="continuous">
    <parent link="upper"/>
    <child link="lower"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <limit velocity="5.0" effort="10.0"/>
  </joint>
</robot>
"#;

    #[test]
    fn xml_round_trips_a_small_document() {
        let xml = parse_xml("<a x=\"1\"><b/><c y=\"2\">text</c></a>").unwrap();
        assert_eq!(xml.tag, "a");
        assert_eq!(xml.attr("x"), Some("1"));
        assert_eq!(xml.children.len(), 2);
        assert_eq!(xml.child("c").unwrap().attr("y"), Some("2"));
    }

    #[test]
    fn xml_handles_comments_and_the_prologue() {
        let xml = parse_xml("<?xml version=\"1.0\"?>\n<!-- hi -->\n<a/>").unwrap();
        assert_eq!(xml.tag, "a");
    }

    #[test]
    fn mismatched_closing_tags_are_rejected() {
        assert!(matches!(
            parse_xml("<a><b></c></a>"),
            Err(XmlError::MismatchedTag { .. })
        ));
    }

    #[test]
    fn entities_are_unescaped_in_attribute_values() {
        let xml = parse_xml("<a x=\"1 &lt; 2 &amp; true\"/>").unwrap();
        assert_eq!(xml.attr("x"), Some("1 < 2 & true"));
    }

    #[test]
    fn imports_a_simple_arm_end_to_end() {
        let imported = import_from_str(SIMPLE_ARM).unwrap();
        assert_eq!(imported.links.len(), 3);
        assert_eq!(imported.joints.len(), 2);
        let base = imported
            .links
            .iter()
            .find(|l| l.name.as_str() == "base")
            .unwrap();
        assert_eq!(base.id, imported.root_link);
        assert!((base.mass.mass.value() - 1.0).abs() < 1e-12);
        assert_eq!(base.collision.len(), 1);

        let shoulder = imported
            .joints
            .iter()
            .find(|j| j.name.as_str() == "shoulder")
            .unwrap();
        assert_eq!(shoulder.kind, JointKind::Revolute);
        assert!((shoulder.origin.translation.z - 0.3).abs() < 1e-12);
        assert_eq!(shoulder.limits.position.unwrap().lo, Radians(-1.57));

        let elbow = imported
            .joints
            .iter()
            .find(|j| j.name.as_str() == "elbow")
            .unwrap();
        assert_eq!(elbow.kind, JointKind::Continuous);
        assert!(elbow.limits.position.is_none());

        let lower = imported
            .links
            .iter()
            .find(|l| l.name.as_str() == "lower")
            .unwrap();
        match &lower.collision[0].geometry {
            CollisionGeometry::Cylinder { radius, length } => {
                assert!((*radius - 0.02).abs() < 1e-12);
                assert!((*length - 0.3).abs() < 1e-12);
            }
            other => panic!("expected a cylinder, got {other:?}"),
        }
    }

    #[test]
    fn a_floating_joint_is_rejected_by_name() {
        let doc = SIMPLE_ARM.replace("type=\"revolute\"", "type=\"floating\"");
        let err = import_from_str(&doc).unwrap_err();
        assert!(matches!(
            err,
            UrdfError::Unsupported { ref construct, .. } if construct == "floating joint"
        ));
    }

    #[test]
    fn a_mimic_joint_is_rejected_by_name() {
        let doc = SIMPLE_ARM.replacen(
            "<limit lower=\"-1.57\" upper=\"1.57\" velocity=\"5.0\" effort=\"20.0\"/>",
            "<limit lower=\"-1.57\" upper=\"1.57\" velocity=\"5.0\" effort=\"20.0\"/><mimic joint=\"elbow\"/>",
            1,
        );
        assert!(matches!(
            import_from_str(&doc),
            Err(UrdfError::Unsupported { ref construct, .. }) if construct == "mimic"
        ));
    }

    #[test]
    fn a_transmission_block_is_rejected_by_name() {
        let doc = SIMPLE_ARM.replace(
            "</robot>",
            "<transmission name=\"t\"></transmission></robot>",
        );
        assert!(matches!(
            import_from_str(&doc),
            Err(UrdfError::Unsupported { ref construct, .. }) if construct == "transmission"
        ));
    }

    #[test]
    fn an_unknown_link_reference_is_reported() {
        let doc = SIMPLE_ARM.replace("link=\"lower\"", "link=\"missing\"");
        assert!(matches!(
            import_from_str(&doc),
            Err(UrdfError::UnknownLink { .. })
        ));
    }

    #[test]
    fn two_disconnected_roots_are_rejected() {
        let doc = SIMPLE_ARM.replace(
            "<joint name=\"shoulder\" type=\"revolute\">",
            "<joint name=\"shoulder\" type=\"fixed\" disabled=\"true\">",
        );
        // Removing the shoulder's effect by detaching is simulated more
        // directly: construct a doc with two independent single-link roots.
        let two_roots = r#"<robot name="r">
          <link name="a"/>
          <link name="b"/>
        </robot>"#;
        let err = import_from_str(two_roots).unwrap_err();
        assert!(
            matches!(err, UrdfError::AmbiguousOrMissingRoot { candidates } if candidates.len() == 2)
        );
        let _ = doc; // the fixed-joint variant above still has one root; not asserted further
    }

    #[test]
    fn duplicate_link_names_are_rejected() {
        let doc = SIMPLE_ARM.replace("name=\"lower\"", "name=\"upper\"");
        assert!(matches!(
            import_from_str(&doc),
            Err(UrdfError::DuplicateName {
                category: "link",
                ..
            })
        ));
    }

    #[test]
    fn a_bounded_joint_without_limits_is_rejected() {
        let doc = SIMPLE_ARM.replace(
            "<limit lower=\"-1.57\" upper=\"1.57\" velocity=\"5.0\" effort=\"20.0\"/>",
            "",
        );
        assert!(matches!(
            import_from_str(&doc),
            Err(UrdfError::Missing {
                element: "limit",
                ..
            })
        ));
    }

    #[test]
    fn rpy_origin_matches_the_extrinsic_fixed_axis_convention() {
        let doc = r#"<robot name="r">
          <link name="a"/>
          <link name="b"/>
          <joint name="j" type="fixed">
            <parent link="a"/>
            <child link="b"/>
            <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
          </joint>
        </robot>"#;
        let imported = import_from_str(doc).unwrap();
        let j = &imported.joints[0];
        let v = j.origin.apply_direction(Vec3::unit_x());
        assert!((v.y - 1.0).abs() < 1e-9, "{v:?}");
    }
}
