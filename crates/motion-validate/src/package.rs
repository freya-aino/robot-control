//! Packaging, per SPECS.md Section 8.2 item 9: "Package accepted references
//! and retain rejected examples with machine-readable reasons."
//!
//! A reference is identified by the content ID of a canonical text form of
//! its numbers (timestamps, base pose and joint values at full `f64`
//! precision, in channel order), so the same motion always gets the same ID
//! and any change to a sample changes it. The curation manifest is TOML in
//! `robot-artifacts`' subset: one `[[accepted]]` or `[[rejected]]` table per
//! reference, a rejected one carrying parallel arrays of reason codes,
//! frames (-1 for a reason with no frame) and messages. Rejected examples
//! are kept in the manifest, not dropped, so a later threshold change can
//! revisit them.

use core::fmt::Write as _;
use robot_artifacts::hash::ContentId;
use robot_artifacts::toml::{self, Table, Value};
use robot_types::motion::{JointValue, MotionRecord, RobotReference};

use crate::report::Verdict;

/// Manifest schema string.
pub const SCHEMA: &str = "motion-curation/1";

/// The content ID of `reference`'s canonical text form.
#[must_use]
pub fn reference_id(reference: &MotionRecord<RobotReference>) -> ContentId {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "robot-reference/1 {} {}",
        reference.skeleton_hash,
        reference.channels.len()
    );
    for c in &reference.channels {
        let _ = writeln!(text, "channel {}", c.name);
    }
    for f in &reference.frames {
        let t = f.root.translation;
        let r = f.root.rotation;
        let _ = write!(
            text,
            "{} {:e} {:e} {:e} {:e} {:e} {:e} {:e}",
            f.timestamp.nanos(),
            t.x,
            t.y,
            t.z,
            r.w,
            r.x,
            r.y,
            r.z
        );
        for v in &f.values {
            let _ = match v {
                JointValue::Scalar { value, .. } => write!(text, " {value:e}"),
                JointValue::Rotation { rotation: q, .. } => {
                    write!(text, " q{:e},{:e},{:e},{:e}", q.w, q.x, q.y, q.z)
                }
                JointValue::Position { position: p } => {
                    write!(text, " p{:e},{:e},{:e}", p.x, p.y, p.z)
                }
            };
        }
        text.push('\n');
    }
    ContentId::of(text.as_bytes())
}

/// One curated reference.
#[derive(Clone, PartialEq, Debug)]
pub struct CurationEntry {
    /// [`reference_id`] of the reference.
    pub reference: ContentId,
    /// Content hash of the human source.
    pub source_hash: String,
    /// Recording session.
    pub session: String,
    /// The verdict.
    pub verdict: Verdict,
}

/// The curation manifest for `entries`, as TOML text.
#[must_use]
pub fn manifest(entries: &[CurationEntry]) -> String {
    let mut root = Table::new();
    root.insert("schema", Value::String(SCHEMA.to_owned()));
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    for e in entries {
        let mut t = Table::new();
        t.insert("reference", Value::String(e.reference.to_string()));
        t.insert("source", Value::String(e.source_hash.clone()));
        t.insert("session", Value::String(e.session.clone()));
        match &e.verdict {
            Verdict::Accepted => accepted.push(Value::Table(t)),
            Verdict::Rejected(reasons) => {
                t.insert(
                    "codes",
                    Value::Array(
                        reasons
                            .iter()
                            .map(|r| Value::String(r.code().to_owned()))
                            .collect(),
                    ),
                );
                t.insert(
                    "frames",
                    Value::Array(
                        reasons
                            .iter()
                            .map(|r| Value::Integer(r.frame().map_or(-1, |f| f as i64)))
                            .collect(),
                    ),
                );
                t.insert(
                    "messages",
                    Value::Array(
                        reasons
                            .iter()
                            .map(|r| Value::String(r.to_string()))
                            .collect(),
                    ),
                );
                rejected.push(Value::Table(t));
            }
        }
    }
    root.insert("accepted", Value::Array(accepted));
    root.insert("rejected", Value::Array(rejected));
    toml::to_string(&root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::RejectionReason;
    use crate::test_support::{g1, standing_reference};

    #[test]
    fn the_id_is_stable_and_sensitive_to_every_sample() {
        let profile = g1();
        let (r, _) = standing_reference(&profile, 3, false);
        assert_eq!(reference_id(&r), reference_id(&r.clone()));
        let mut changed = r.clone();
        if let JointValue::Scalar { value, .. } = &mut changed.frames[2].values[5] {
            *value += 1e-12;
        }
        assert_ne!(reference_id(&r), reference_id(&changed));
    }

    #[test]
    fn the_manifest_round_trips_and_keeps_rejections_with_codes() {
        let id = ContentId::of(b"x");
        let entries = [
            CurationEntry {
                reference: id.clone(),
                source_hash: "sha256:a".into(),
                session: "s1".into(),
                verdict: Verdict::Accepted,
            },
            CurationEntry {
                reference: id,
                source_hash: "sha256:b".into(),
                session: "s2".into(),
                verdict: Verdict::Rejected(vec![
                    RejectionReason::LimitViolation { frame: 4 },
                    RejectionReason::TrackingMissing,
                ]),
            },
        ];
        let text = manifest(&entries);
        let parsed = toml::parse(&text).unwrap();
        assert_eq!(parsed.require_str("schema").unwrap(), SCHEMA);
        let rejected = parsed.require_array("rejected").unwrap();
        let Value::Table(t) = &rejected[0] else {
            panic!()
        };
        let codes: Vec<_> = t
            .require_array("codes")
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(codes, ["limit_violation", "tracking_missing"]);
        assert_eq!(t.require_array("frames").unwrap()[1], Value::Integer(-1));
        assert_eq!(parsed.require_array("accepted").unwrap().len(), 1);
    }
}
