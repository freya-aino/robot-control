//! Non-fatal findings about a composed stage.
//!
//! A layer can reference content that exists only in another application, for
//! example an Omniverse MDL material. Such content does not change geometry,
//! physics or joints, so it is reported as a [`Warning`] and loading continues.
//! Anything that would change the robot model is an error elsewhere.

use crate::stage::{Prim, Stage};
use crate::value::Value;

/// A finding that does not stop loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    /// The prim the finding is about.
    pub path: String,
    /// What was found and what it means.
    pub message: String,
}

/// Warnings for shaders whose source is an MDL file. MDL materials run only in
/// Omniverse; they are not resolved here.
#[must_use]
pub fn shader_warnings(stage: &Stage) -> Vec<Warning> {
    let mut out = Vec::new();
    stage.for_each_prim(|prim: &Prim| {
        if prim.type_name.as_deref() != Some("Shader") {
            return;
        }
        let Some(source) = prim
            .property("info:mdl:sourceAsset")
            .and_then(|p| p.default.as_ref())
        else {
            return;
        };
        let file = match source {
            Value::AssetPath(path) => path.clone(),
            _ => return,
        };
        if file.ends_with(".mdl") {
            out.push(Warning {
                path: prim.path.clone(),
                message: format!(
                    "MDL shader source {file} is available only in Omniverse; the material is not resolved. Geometry, joints and physics are unaffected."
                ),
            });
        }
    });
    out
}
