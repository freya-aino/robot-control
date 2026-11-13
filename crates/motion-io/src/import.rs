//! Format detection, inspection and import with immutable originals, per
//! SPECS.md Section 8.2 items 1 and 2: "Inspect format, license, skeleton,
//! rates and coordinate conventions" and "Preserve immutable originals and
//! assign content IDs".
//!
//! [`import`] stores the original bytes in a `robot-artifacts`
//! [`BlobStore`] *before* parsing, so the record's `source_hash` is the
//! content ID of bytes that are already on disk and can be re-read, and a
//! record can never cite an original that was not kept. [`inspect`]
//! summarizes a parsed record, including two guesses (up axis and length
//! unit) that are labelled as heuristics: a file that does not state its
//! conventions cannot have them read, only estimated, and Section 8.2 item
//! 3's correction step takes them as explicit inputs.

use core::fmt;
use robot_artifacts::hash::ContentId;
use robot_artifacts::store::{BlobStore, StoreError};
use robot_types::frames::Vec3;
use robot_types::motion::{
    BodyModelParameters, HumanSkeleton, JointValue, MotionKind, MotionRecord, SourceLicense,
    Validity,
};

use crate::amass::{self, AmassError, AmassImport};
use crate::bvh::{self, BvhError};
use crate::c3d::{self, C3dError, C3dImport};

/// A recognized file format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    /// Biovision hierarchy text.
    Bvh,
    /// C3D binary.
    C3d,
    /// NumPy archive, read as AMASS.
    AmassNpz,
}

/// Recognizes a format by content, not by file name.
#[must_use]
pub fn detect(bytes: &[u8]) -> Option<Format> {
    let text_start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map_or(&[][..], |i| &bytes[i..]);
    if text_start.starts_with(b"HIERARCHY") {
        Some(Format::Bvh)
    } else if bytes.starts_with(b"PK\x03\x04") {
        Some(Format::AmassNpz)
    } else if bytes.len() >= 512 && bytes[1] == 0x50 {
        Some(Format::C3d)
    } else {
        None
    }
}

/// What an import produced.
#[derive(Clone, PartialEq, Debug)]
pub enum Imported {
    /// A BVH skeleton record.
    Bvh(Box<MotionRecord<HumanSkeleton>>),
    /// A C3D marker record with residuals and analog channels.
    C3d(Box<C3dImport>),
    /// AMASS body-model parameters.
    Amass(Box<AmassImport>),
}

/// Why an import failed.
#[derive(Debug)]
pub enum ImportError {
    /// The format was not recognized.
    UnknownFormat,
    /// The original could not be stored.
    Store(StoreError),
    /// BVH parse failure.
    Bvh(BvhError),
    /// C3D parse failure.
    C3d(C3dError),
    /// AMASS parse failure.
    Amass(AmassError),
    /// BVH must be UTF-8 text.
    NotText,
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFormat => f.write_str("unrecognized motion format"),
            Self::Store(e) => write!(f, "storing the original: {e}"),
            Self::Bvh(e) => write!(f, "{e}"),
            Self::C3d(e) => write!(f, "{e}"),
            Self::Amass(e) => write!(f, "{e}"),
            Self::NotText => f.write_str("BVH file is not UTF-8 text"),
        }
    }
}

impl core::error::Error for ImportError {}

/// Stores `bytes` in `store`, then parses them; the record's `source_hash`
/// is the stored original's content ID.
pub fn import(
    store: &BlobStore,
    bytes: &[u8],
    license: SourceLicense,
) -> Result<(ContentId, Imported), ImportError> {
    let format = detect(bytes).ok_or(ImportError::UnknownFormat)?;
    let id = store.put(bytes).map_err(ImportError::Store)?;
    let hash = id.to_string();
    let imported = match format {
        Format::Bvh => {
            let text = core::str::from_utf8(bytes).map_err(|_| ImportError::NotText)?;
            Imported::Bvh(Box::new(
                bvh::parse(text, &hash, license).map_err(ImportError::Bvh)?,
            ))
        }
        Format::C3d => Imported::C3d(Box::new(
            c3d::parse(bytes, &hash, license).map_err(ImportError::C3d)?,
        )),
        Format::AmassNpz => Imported::Amass(Box::new(
            amass::parse(bytes, &hash, license).map_err(ImportError::Amass)?,
        )),
    };
    Ok((id, imported))
}

/// A coordinate axis, for the up-axis guess.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UpAxis {
    /// +X.
    X,
    /// +Y.
    Y,
    /// +Z.
    Z,
}

/// A summary of a record, per SPECS.md Section 8.2 item 1.
#[derive(Clone, PartialEq, Debug)]
pub struct Inspection {
    /// Frames.
    pub frames: usize,
    /// Channels.
    pub channels: usize,
    /// Mean rate, Hz.
    pub rate: Option<f64>,
    /// Duration, seconds.
    pub duration: f64,
    /// Fraction of samples marked valid.
    pub valid_fraction: f64,
    /// Licence identifier.
    pub license: String,
    /// Whether the licence permits training.
    pub training_permitted: bool,
    /// Heuristic: the axis the root (or marker cloud) sits highest and
    /// steadiest along. `None` with no translation to judge from.
    pub up_axis_guess: Option<UpAxis>,
    /// Heuristic: the length unit that makes the figure's extent along the
    /// guessed up axis human-sized (`"m"`, `"cm"` or `"mm"`).
    pub length_unit_guess: Option<&'static str>,
}

/// Inspects a record.
#[must_use]
pub fn inspect<K: MotionKind>(record: &MotionRecord<K>) -> Inspection {
    // Points that describe the figure: the root, and every marker sample.
    let mut points: Vec<Vec3> = record.frames.iter().map(|f| f.root.translation).collect();
    for f in &record.frames {
        for (v, ok) in f.values.iter().zip(&f.validity) {
            if let (JointValue::Position { position }, Validity::Valid) = (v, ok) {
                points.push(*position);
            }
        }
    }
    let up = up_axis(&points);
    let extent = up.and_then(|axis| {
        let k = axis as usize;
        // Rest-pose extent from offsets when the record has a skeleton,
        // else the marker cloud's extent.
        let along = |v: Vec3| v.to_xyz()[k];
        let skeleton: f64 = record.channels.iter().map(|c| along(c.offset).abs()).sum();
        if skeleton > 0.0 {
            Some(skeleton)
        } else {
            let (lo, hi) = points
                .iter()
                .map(|p| along(*p))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                });
            (hi > lo).then_some(hi - lo)
        }
    });
    let unit = extent.and_then(|e| match e {
        e if (0.5..=3.0).contains(&e) => Some("m"),
        e if (50.0..=300.0).contains(&e) => Some("cm"),
        e if (500.0..=3000.0).contains(&e) => Some("mm"),
        _ => None,
    });
    Inspection {
        frames: record.frame_count(),
        channels: record.channels.len(),
        rate: record.mean_rate().map(|r| r.value()),
        duration: record.duration().as_secs_f64(),
        valid_fraction: record.valid_fraction(),
        license: record.source_license.identifier.clone(),
        training_permitted: record.source_license.permits_training(),
        up_axis_guess: up,
        length_unit_guess: unit,
    }
}

fn up_axis(points: &[Vec3]) -> Option<UpAxis> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let score = |k: usize| {
        let vals: Vec<f64> = points.iter().map(|p| p.to_xyz()[k]).collect();
        let mean = vals.iter().sum::<f64>() / n;
        let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        mean / (std + 1e-9 * mean.abs().max(1.0))
    };
    let s = [score(0), score(1), score(2)];
    let best = (0..3).max_by(|&a, &b| s[a].total_cmp(&s[b]))?;
    (s[best] > 0.0).then_some([UpAxis::X, UpAxis::Y, UpAxis::Z][best])
}

/// Inspects an imported file.
#[must_use]
pub fn inspect_imported(imported: &Imported) -> Inspection {
    match imported {
        Imported::Bvh(r) => inspect(r),
        Imported::C3d(c) => inspect(&c.record),
        Imported::Amass(a) => inspect::<BodyModelParameters>(&a.record),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_artifacts::store::StoreQuota;

    const BVH: &str = "HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 60 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
  JOINT Leg
  {
    OFFSET 10 -90 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 -5 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0 92 0 0 0 0 0 0 0 0 0 0
5 93 1 0 0 0 0 0 0 0 0 0
10 92 2 0 0 0 0 0 0 0 0 0
";

    fn store() -> (BlobStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("motion-io-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (BlobStore::open(&dir, StoreQuota::default()).unwrap(), dir)
    }

    #[test]
    fn formats_are_detected_by_content() {
        assert_eq!(detect(BVH.as_bytes()), Some(Format::Bvh));
        assert_eq!(detect(b"PK\x03\x04rest"), Some(Format::AmassNpz));
        assert_eq!(detect(b"not motion"), None);
    }

    #[test]
    fn the_original_is_stored_before_parsing_and_cited_by_the_record() {
        let (store, dir) = store();
        let (id, imported) = import(
            &store,
            BVH.as_bytes(),
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        let Imported::Bvh(record) = &imported else {
            panic!()
        };
        assert_eq!(record.source_hash, id.to_string());
        assert_eq!(store.get(&id).unwrap(), BVH.as_bytes());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_y_up_centimetre_bvh_is_guessed_as_such() {
        let record = bvh::parse(BVH, "x", SourceLicense::unaudited("fixture-internal")).unwrap();
        let i = inspect(&record);
        assert_eq!(i.up_axis_guess, Some(UpAxis::Y));
        assert_eq!(i.length_unit_guess, Some("cm"));
        assert_eq!(i.frames, 3);
        assert!(!i.training_permitted);
    }
}
