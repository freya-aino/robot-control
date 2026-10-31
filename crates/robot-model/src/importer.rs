//! The plug-in seam for model formats: anything that turns a robot description
//! document into [`ImportedKinematics`] can feed the model pipeline.
//!
//! [`UrdfImporter`] is the first implementor and wraps the existing
//! [`crate::urdf`] loader without changing it. A new format (a vendor
//! JSON description) is added as another implementor in its own crate, with no
//! edit to this one.

use crate::urdf::{self, ImportedKinematics, UrdfError};

/// Converts a robot description document into imported kinematics.
pub trait ModelImporter {
    /// Why the document could not be imported.
    type Error;

    /// Imports the kinematic structure described by `text`.
    fn import(&self, text: &str) -> Result<ImportedKinematics, Self::Error>;
}

/// Imports URDF documents through [`crate::urdf::import_from_str`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UrdfImporter;

impl ModelImporter for UrdfImporter {
    type Error = UrdfError;

    fn import(&self, text: &str) -> Result<ImportedKinematics, UrdfError> {
        urdf::import_from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urdf_importer_matches_the_direct_loader_on_invalid_input() {
        let via_trait = UrdfImporter.import("<not-a-robot").unwrap_err();
        let direct = urdf::import_from_str("<not-a-robot").unwrap_err();
        assert_eq!(format!("{via_trait:?}"), format!("{direct:?}"));
    }

    #[test]
    fn urdf_importer_imports_the_g1_fixture_like_the_direct_loader() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/robots/unitree_g1/g1_29dof_rev_1_0.urdf"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let via_trait = UrdfImporter.import(&text).unwrap();
        let direct = urdf::import_from_str(&text).unwrap();
        assert!(!via_trait.links.is_empty());
        assert_eq!(via_trait.links.len(), direct.links.len());
        assert_eq!(via_trait.joints.len(), direct.joints.len());
    }

    #[test]
    fn a_format_importer_can_be_written_outside_the_urdf_module() {
        struct Rejects;
        impl ModelImporter for Rejects {
            type Error = &'static str;
            fn import(&self, _text: &str) -> Result<ImportedKinematics, &'static str> {
                Err("unsupported format")
            }
        }
        fn import_with<I: ModelImporter>(
            i: &I,
            text: &str,
        ) -> Result<ImportedKinematics, I::Error> {
            i.import(text)
        }
        assert_eq!(
            import_with(&Rejects, "x").unwrap_err(),
            "unsupported format"
        );
    }
}
