//! `robot motion validate <source>` — a real subset of PLAN.md §4's
//! proposed `robot motion validate <reference-id> --config <path>`.
//!
//! This parses a BVH source file and runs
//! `robot_types::motion::MotionRecord::validate`, SPECS.md §8's check on
//! the *source* motion record itself (channel widths, frame rate,
//! license, missing/interpolated-frame accounting). It is not the full
//! `motion_validate::report::ValidationReport` against a *retargeted*
//! robot motion — that needs a skill-specific source-to-robot joint
//! mapping this workspace has no generic config format for yet, so
//! `robot motion retarget` is not wired to a real solve in this revision.

use super::EXIT_FAILURE;
use robot_artifacts::hash::ContentId;
use robot_types::motion::SourceLicense;
use std::path::Path;

/// Runs `robot motion validate <path>`. Returns the process exit code.
pub fn validate(path: &Path) -> i32 {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: could not read {}: {e}", path.display());
            return EXIT_FAILURE;
        }
    };
    let text = match std::str::from_utf8(&bytes) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {} is not valid UTF-8: {e}", path.display());
            return EXIT_FAILURE;
        }
    };

    let source_hash = ContentId::of(&bytes).to_string();
    let license = SourceLicense::unaudited(format!("cli-import:{}", path.display()));

    let record = match motion_io::bvh::parse(text, &source_hash, license) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: could not parse {} as BVH: {e}", path.display());
            return EXIT_FAILURE;
        }
    };

    println!("{}: BVH motion record", path.display());
    println!("  frames:          {}", record.frame_count());
    println!("  duration:        {}", record.duration());
    match record.mean_rate() {
        Some(rate) => println!("  mean rate:       {rate:?}"),
        None => println!("  mean rate:       (undefined; fewer than 2 frames)"),
    }
    println!("  valid fraction:  {:.4}", record.valid_fraction());

    match record.validate() {
        Ok(()) => {
            println!("  result:          ACCEPTED");
            0
        }
        Err(e) => {
            println!("  result:          REJECTED: {e}");
            EXIT_FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const BVH: &str = "HIERARCHY\nROOT Hips\n{\n\tOFFSET 0.00 0.00 0.00\n\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n}\nMOTION\nFrames: 2\nFrame Time: 0.033333\n0.0 0.0 0.0 0.0 0.0 0.0\n0.0 0.0 0.0 1.0 0.0 0.0\n";

    fn write_temp(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "robot-cli-test-motion-{label}-{}",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn validating_a_well_formed_bvh_file_succeeds() {
        let path = write_temp("well-formed", BVH.as_bytes());
        let code = validate(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn validating_a_non_bvh_file_fails_clearly() {
        let path = write_temp("non-bvh", b"not a bvh file at all");
        let code = validate(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(code, EXIT_FAILURE);
    }
}
