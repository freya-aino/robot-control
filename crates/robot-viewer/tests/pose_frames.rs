//! Golden-file and round-trip tests for the pose frame dump and SVG render.
//!
//! The golden file `tests/golden/frame_fixture.svg` is the pinned output of
//! `svg::render` for [`fixture`]. To regenerate it after an intended change,
//! run `ROBOT_VIEWER_UPDATE_GOLDEN=1 cargo test -p robot-viewer --test pose_frames`
//! and review the diff. A missing or different golden file fails the test.

use robot_types::time::{ClockId, Timestamp};
use robot_viewer::pose::{self, FrameDump, PoseError, PoseFrame};
use robot_viewer::{csv, jsonl, svg};
use std::fs;
use std::path::{Path, PathBuf};

const GOLDEN: &str = "tests/golden/frame_fixture.svg";

fn fixture() -> PoseFrame {
    PoseFrame {
        step: 7,
        timestamp: Timestamp::sampling(700_000, ClockId::Simulation),
        pelvis_position: [0.1, 0.0, 0.79],
        pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
        joint_angles: vec![0.25, -0.5, 2.0],
    }
}

fn joints() -> Vec<String> {
    vec![
        "left_hip_pitch_joint".to_owned(),
        "left_knee_joint".to_owned(),
        "left_ankle_pitch_joint".to_owned(),
    ]
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "robot-viewer-pose-tests-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn frame_at(step: u64) -> PoseFrame {
    PoseFrame {
        step,
        timestamp: Timestamp::sampling(step as i64 * 100_000, ClockId::Simulation),
        pelvis_position: [0.001 * step as f64, 0.0, 0.79],
        pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
        joint_angles: vec![0.01 * step as f64, -0.02, 0.5],
    }
}

fn write_run(dir: &Path, steps: u64) -> Result<u64, PoseError> {
    let mut dump = FrameDump::create(dir, &joints())?;
    for step in 1..=steps {
        dump.push(&frame_at(step))?;
    }
    dump.finish()
}

#[test]
fn svg_matches_the_golden_file() {
    let rendered = svg::render(&fixture());
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN);
    if std::env::var_os("ROBOT_VIEWER_UPDATE_GOLDEN").is_some() {
        fs::write(&path, &rendered).expect("write golden file");
    }
    let golden = fs::read_to_string(&path).expect("golden file exists");
    assert_eq!(rendered, golden, "SVG output drifted from the golden file");
}

#[test]
fn svg_is_one_document_per_frame_with_a_step_label() {
    let svg_text = svg::render(&frame_at(42));
    assert_eq!(svg_text.matches("<svg ").count(), 1);
    assert!(svg_text.contains("step 42 t="));
}

#[test]
fn a_repeated_run_writes_byte_identical_files() {
    let a = scratch("repeat-a");
    let b = scratch("repeat-b");
    write_run(&a, 25).unwrap();
    write_run(&b, 25).unwrap();
    for file in [pose::CSV_FILE, pose::JSONL_FILE] {
        assert_eq!(
            fs::read(a.join(file)).unwrap(),
            fs::read(b.join(file)).unwrap(),
            "{file} differs between identical runs"
        );
    }
    for step in 1..=25 {
        let name = pose::svg_file_name(step);
        assert_eq!(
            fs::read(a.join(pose::SVG_DIR).join(&name)).unwrap(),
            fs::read(b.join(pose::SVG_DIR).join(&name)).unwrap(),
            "{name} differs between identical runs"
        );
    }
    let _ = fs::remove_dir_all(&a);
    let _ = fs::remove_dir_all(&b);
}

#[test]
fn csv_and_jsonl_read_back_to_the_same_frames() {
    let dir = scratch("readback");
    write_run(&dir, 12).unwrap();
    let from_csv = csv::read_file(&dir.join(pose::CSV_FILE)).unwrap();
    let from_jsonl = jsonl::read_file(&dir.join(pose::JSONL_FILE)).unwrap();
    assert_eq!(from_csv.frames.len(), 12);
    assert_eq!(from_csv.frames, from_jsonl.frames);
    assert_eq!(from_csv.joints, joints());
    let timeline = from_jsonl.into_timeline().unwrap();
    assert_eq!(timeline.len(), 12);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_unwritable_output_directory_fails_before_any_step() {
    let dir = scratch("unwritable");
    fs::create_dir_all(&dir).unwrap();
    // A regular file where the svg directory should be makes the create fail.
    fs::write(dir.join(pose::SVG_DIR), b"not a directory").unwrap();
    let result = FrameDump::create(&dir, &joints());
    assert!(matches!(result, Err(PoseError::Io { .. })));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_dropped_dump_removes_its_partial_output() {
    let dir = scratch("drop-cleanup");
    {
        let mut dump = FrameDump::create(&dir, &joints()).unwrap();
        dump.push(&frame_at(1)).unwrap();
        dump.push(&frame_at(2)).unwrap();
        // No finish(): simulates an early error return from the caller.
    }
    assert!(!dir.join(pose::CSV_FILE).exists());
    assert!(!dir.join(pose::JSONL_FILE).exists());
    assert!(!dir
        .join(pose::SVG_DIR)
        .join(pose::svg_file_name(1))
        .exists());
    assert!(!dir
        .join(pose::SVG_DIR)
        .join(pose::svg_file_name(2))
        .exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_truncated_jsonl_file_is_rejected_with_no_partial_load() {
    let dir = scratch("truncated");
    write_run(&dir, 6).unwrap();
    let path = dir.join(pose::JSONL_FILE);
    let text = fs::read_to_string(&path).unwrap();
    let cut = &text[..text.len() - 9];
    fs::write(&path, cut).unwrap();
    assert!(jsonl::read_file(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_line_boundary_truncated_jsonl_file_is_rejected_with_no_partial_load() {
    let dir = scratch("line-truncated-jsonl");
    write_run(&dir, 6).unwrap();
    let path = dir.join(pose::JSONL_FILE);
    let text = fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    // Drop the footer line
    let cut = lines[..lines.len() - 1].join("\n") + "\n";
    fs::write(&path, cut).unwrap();
    assert!(jsonl::read_file(&path).is_err());

    // Drop last 2 frames and footer
    let cut_shorter = lines[..lines.len() - 3].join("\n") + "\n";
    fs::write(&path, cut_shorter).unwrap();
    assert!(jsonl::read_file(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupt_csv_file_is_rejected_with_no_partial_load() {
    let dir = scratch("corrupt-csv");
    write_run(&dir, 6).unwrap();
    let path = dir.join(pose::CSV_FILE);
    let text = fs::read_to_string(&path).unwrap();
    let corrupt = text.replacen("0.79", "NaN", 1);
    fs::write(&path, corrupt).unwrap();
    assert!(csv::read_file(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_line_boundary_truncated_csv_file_is_rejected_with_no_partial_load() {
    let dir = scratch("line-truncated-csv");
    write_run(&dir, 6).unwrap();
    let path = dir.join(pose::CSV_FILE);
    let text = fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    // Drop the footer line
    let cut = lines[..lines.len() - 1].join("\n") + "\n";
    fs::write(&path, cut).unwrap();
    assert!(csv::read_file(&path).is_err());

    // Drop last 2 frames and footer
    let cut_shorter = lines[..lines.len() - 3].join("\n") + "\n";
    fs::write(&path, cut_shorter).unwrap();
    assert!(csv::read_file(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_csv_file_with_wrong_schema_marker_is_rejected() {
    let dir = scratch("wrong-marker-csv");
    write_run(&dir, 4).unwrap();
    let path = dir.join(pose::CSV_FILE);
    let text = fs::read_to_string(&path).unwrap();
    let corrupt = text.replacen(csv::SCHEMA_MARKER, "# robot-viewer.pose-frames v2", 1);
    fs::write(&path, corrupt).unwrap();
    assert!(csv::read_file(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn wrong_footer_step_counts_are_rejected() {
    let dir = scratch("wrong-footer");
    write_run(&dir, 4).unwrap();

    let jsonl_path = dir.join(pose::JSONL_FILE);
    let jsonl_text = fs::read_to_string(&jsonl_path).unwrap();
    fs::write(
        &jsonl_path,
        jsonl_text.replace("\"steps\":4", "\"steps\":5"),
    )
    .unwrap();
    assert!(jsonl::read_file(&jsonl_path).is_err());

    let csv_path = dir.join(pose::CSV_FILE);
    let csv_text = fs::read_to_string(&csv_path).unwrap();
    fs::write(
        &csv_path,
        csv_text.replace("# end steps=4", "# end steps=5"),
    )
    .unwrap();
    assert!(csv::read_file(&csv_path).is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rerunning_into_the_same_directory_removes_stale_svgs() {
    let dir = scratch("stale");
    write_run(&dir, 10).unwrap();
    write_run(&dir, 4).unwrap();
    let svg_dir = dir.join(pose::SVG_DIR);
    assert!(svg_dir.join(pose::svg_file_name(4)).exists());
    assert!(!svg_dir.join(pose::svg_file_name(5)).exists());
    assert!(!svg_dir.join(pose::svg_file_name(10)).exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_output_path_is_rejected_before_anything_is_written() {
    let result = FrameDump::create(Path::new(""), &joints());
    assert!(matches!(result, Err(PoseError::EmptyDirectory)));
}

#[test]
fn a_committed_run_loads_back_through_the_manifest() {
    let dir = scratch("committed-load");
    write_run(&dir, 5).unwrap();
    assert!(dir.join(pose::COMMIT_FILE).is_file());
    let log = pose::load_committed(&dir).unwrap();
    assert_eq!(log.frames.len(), 5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_uncommitted_dump_is_not_loadable_and_leaves_no_staging_behind() {
    let dir = scratch("uncommitted");
    {
        let mut dump = FrameDump::create(&dir, &joints()).unwrap();
        dump.push(&frame_at(1)).unwrap();
    }
    assert!(!dir.join(pose::COMMIT_FILE).exists());
    assert!(matches!(
        pose::load_committed(&dir),
        Err(PoseError::NotCommitted { .. })
    ));
    assert!(
        !dir.join(".pose-staging").exists(),
        "staging directory was not removed on drop"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_dropped_dump_leaves_an_earlier_committed_run_untouched() {
    let dir = scratch("drop-keeps-committed");
    write_run(&dir, 4).unwrap();
    let before = fs::read(dir.join(pose::CSV_FILE)).unwrap();
    {
        let mut dump = FrameDump::create(&dir, &joints()).unwrap();
        dump.push(&frame_at(1)).unwrap();
        dump.push(&frame_at(2)).unwrap();
    }
    assert_eq!(fs::read(dir.join(pose::CSV_FILE)).unwrap(), before);
    let log = pose::load_committed(&dir).unwrap();
    assert_eq!(log.frames.len(), 4);
    assert!(dir
        .join(pose::SVG_DIR)
        .join(pose::svg_file_name(4))
        .exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_rejected_create_leaves_an_earlier_committed_run_untouched() {
    let dir = scratch("bad-create-keeps-committed");
    write_run(&dir, 3).unwrap();
    let bad_joints = vec!["bad name\n".to_owned()];
    assert!(matches!(
        FrameDump::create(&dir, &bad_joints),
        Err(PoseError::InvalidJointName { .. })
    ));
    let log = pose::load_committed(&dir).unwrap();
    assert_eq!(log.frames.len(), 3);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_committed_run_without_its_manifest_is_not_loadable() {
    let dir = scratch("no-manifest");
    write_run(&dir, 3).unwrap();
    fs::remove_file(dir.join(pose::COMMIT_FILE)).unwrap();
    assert!(matches!(
        pose::load_committed(&dir),
        Err(PoseError::NotCommitted { .. })
    ));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_committed_run_with_an_svg_past_its_last_step_is_rejected() {
    let dir = scratch("extra-svg");
    write_run(&dir, 3).unwrap();
    fs::write(
        dir.join(pose::SVG_DIR).join(pose::svg_file_name(4)),
        "<svg/>",
    )
    .unwrap();
    assert!(pose::load_committed(&dir).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_committed_run_with_a_missing_svg_is_rejected() {
    let dir = scratch("missing-svg");
    write_run(&dir, 3).unwrap();
    fs::remove_file(dir.join(pose::SVG_DIR).join(pose::svg_file_name(2))).unwrap();
    assert!(pose::load_committed(&dir).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_commit_manifest_with_a_wrong_marker_or_count_is_rejected() {
    let dir = scratch("bad-manifest");
    write_run(&dir, 3).unwrap();
    let manifest = dir.join(pose::COMMIT_FILE);
    let good = fs::read_to_string(&manifest).unwrap();

    fs::write(&manifest, good.replacen("v1", "v2", 1)).unwrap();
    assert!(pose::load_committed(&dir).is_err());

    fs::write(&manifest, good.replace("steps 3", "steps 4")).unwrap();
    assert!(pose::load_committed(&dir).is_err());

    fs::write(&manifest, good.replace("steps 3\n", "steps 3")).unwrap();
    assert!(pose::load_committed(&dir).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_second_commit_replaces_the_first_run_completely() {
    let dir = scratch("recommit");
    write_run(&dir, 6).unwrap();
    write_run(&dir, 2).unwrap();
    let log = pose::load_committed(&dir).unwrap();
    assert_eq!(log.frames.len(), 2);
    assert!(!dir
        .join(pose::SVG_DIR)
        .join(pose::svg_file_name(3))
        .exists());
    let _ = fs::remove_dir_all(&dir);
}

fn committed_bytes(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    for name in [pose::CSV_FILE, pose::JSONL_FILE, pose::COMMIT_FILE] {
        files.push((name.to_owned(), fs::read(dir.join(name)).unwrap()));
    }
    for step in 1..=3 {
        let name = pose::svg_file_name(step);
        let path = dir.join(pose::SVG_DIR).join(&name);
        files.push((name, fs::read(path).unwrap()));
    }
    files
}

#[test]
fn a_second_create_is_rejected_while_the_first_holds_the_directory() {
    let dir = scratch("exclusive");
    let first = FrameDump::create(&dir, &joints()).unwrap();
    let second = FrameDump::create(&dir, &joints());
    assert!(matches!(second, Err(PoseError::Locked { .. })));
    drop(first);
    assert!(!dir.join(".pose-staging").exists());
    FrameDump::create(&dir, &joints()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_rejected_second_create_does_not_remove_the_live_lock() {
    let dir = scratch("exclusive-keeps-lock");
    let mut first = FrameDump::create(&dir, &joints()).unwrap();
    assert!(FrameDump::create(&dir, &joints()).is_err());
    assert!(dir.join(".pose-staging").is_dir());
    first.push(&frame_at(1)).unwrap();
    assert_eq!(first.finish().unwrap(), 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_new_writers_staging_survives_after_the_previous_finish() {
    let dir = scratch("finished-drop-keeps-next-stage");
    let mut first = FrameDump::create(&dir, &joints()).unwrap();
    first.push(&frame_at(1)).unwrap();
    assert_eq!(first.finish().unwrap(), 1);

    let mut second = FrameDump::create(&dir, &joints()).unwrap();
    let mut second_frame = frame_at(1);
    second_frame.pelvis_position[0] = 9.0;
    second.push(&second_frame).unwrap();

    let staging = dir.join(".pose-staging");
    assert!(staging.is_dir());
    assert!(staging.join(pose::CSV_FILE).is_file());
    assert!(staging.join(pose::JSONL_FILE).is_file());
    assert!(staging.join("owner").is_file());
    assert_eq!(second.finish().unwrap(), 1);

    let committed = pose::load_committed(&dir).unwrap();
    assert_eq!(committed.frames.len(), 1);
    assert_eq!(committed.frames[0].pelvis_position[0], 9.0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn concurrent_create_and_finish_only_refuses_contenders_as_locked() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    const THREADS: usize = 3;
    const ITERATIONS: usize = 200;

    let dir = Arc::new(scratch("concurrent-finish"));
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut workers = Vec::new();

    for _ in 0..THREADS {
        let dir = Arc::clone(&dir);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            barrier.wait();
            let mut committed = 0;
            for _ in 0..ITERATIONS {
                match FrameDump::create(&dir, &joints()) {
                    Ok(mut dump) => {
                        dump.push(&frame_at(1)).unwrap();
                        assert_eq!(dump.finish().unwrap(), 1);
                        committed += 1;
                    }
                    Err(PoseError::Locked { .. }) => {}
                    Err(error) => panic!("unexpected create error under contention: {error}"),
                }
            }
            committed
        }));
    }

    let committed = workers
        .into_iter()
        .map(|worker| worker.join().expect("writer thread panicked"))
        .sum::<usize>();
    assert!(committed > 0, "no writer committed a run");

    let log = pose::load_committed(&dir).unwrap();
    assert_eq!(log.frames.len(), 1);
    assert_eq!(log.frames[0].step, 1);
    let _ = fs::remove_dir_all(&*dir);
}

#[test]
fn a_failed_finish_keeps_the_committed_run_and_releases_the_lock() {
    let dir = scratch("failed-finish");
    write_run(&dir, 1).unwrap();
    let snapshot = |dir: &Path| -> Vec<(String, Vec<u8>)> {
        [pose::CSV_FILE, pose::JSONL_FILE, pose::COMMIT_FILE]
            .iter()
            .map(|name| ((*name).to_owned(), fs::read(dir.join(name)).unwrap()))
            .chain(std::iter::once((
                pose::svg_file_name(1),
                fs::read(dir.join(pose::SVG_DIR).join(pose::svg_file_name(1))).unwrap(),
            )))
            .collect()
    };
    let before = snapshot(&dir);
    let mut dump = FrameDump::create(&dir, &joints()).unwrap();
    dump.push(&frame_at(1)).unwrap();
    dump.push(&frame_at(2)).unwrap();
    let blocker = dir.join(pose::SVG_DIR).join(pose::svg_file_name(2));
    fs::create_dir_all(&blocker).unwrap();
    assert!(dump.finish().is_err());
    assert_eq!(snapshot(&dir), before);
    assert!(!dir.join(".pose-staging").exists());
    fs::remove_dir(&blocker).unwrap();
    FrameDump::create(&dir, &joints()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(target_os = "linux")]
fn owner_record(pid: u32, boot: &str) -> String {
    let namespace = fs::read_link("/proc/self/ns/pid").unwrap();
    format!(
        "pid {pid}\nnamespace {}\nboot {boot}\n",
        namespace.display()
    )
}

#[cfg(target_os = "linux")]
fn current_boot() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .trim()
        .to_owned()
}

#[test]
#[cfg(target_os = "linux")]
fn a_stale_staging_directory_of_a_dead_run_is_reclaimed() {
    let dir = scratch("stale-reclaim");
    let stale = dir.join(".pose-staging");
    fs::create_dir_all(stale.join(pose::SVG_DIR)).unwrap();
    fs::write(
        stale.join("owner"),
        owner_record(2147483646, &current_boot()),
    )
    .unwrap();
    fs::write(stale.join(pose::CSV_FILE), "left over\n").unwrap();
    let mut dump = FrameDump::create(&dir, &joints()).unwrap();
    dump.push(&frame_at(1)).unwrap();
    assert_eq!(dump.finish().unwrap(), 1);
    assert!(!dir.join(".pose-staging").exists());
    assert_eq!(pose::load_committed(&dir).unwrap().frames.len(), 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg(target_os = "linux")]
fn a_stale_owner_from_another_boot_is_refused_and_kept() {
    let dir = scratch("other-boot");
    let staging = dir.join(".pose-staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(
        staging.join("owner"),
        owner_record(2147483646, "00000000-0000-0000-0000-000000000000"),
    )
    .unwrap();
    fs::write(staging.join(pose::CSV_FILE), "kept\n").unwrap();
    let result = FrameDump::create(&dir, &joints());
    assert!(matches!(result, Err(PoseError::Locked { .. })));
    assert_eq!(
        fs::read_to_string(staging.join(pose::CSV_FILE)).unwrap(),
        "kept\n"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_bare_pid_owner_without_namespace_is_refused_and_kept() {
    let dir = scratch("bare-pid");
    let staging = dir.join(".pose-staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("owner"), "pid 2147483646\n").unwrap();
    let result = FrameDump::create(&dir, &joints());
    assert!(matches!(result, Err(PoseError::Locked { .. })));
    assert!(staging.join("owner").is_file());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_staging_directory_with_a_live_owner_is_refused_and_kept() {
    let dir = scratch("live-owner");
    let staging = dir.join(".pose-staging");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("owner"), "pid 1\n").unwrap();
    fs::write(staging.join(pose::CSV_FILE), "in progress\n").unwrap();
    let result = FrameDump::create(&dir, &joints());
    assert!(matches!(result, Err(PoseError::Locked { .. })));
    assert_eq!(
        fs::read_to_string(staging.join(pose::CSV_FILE)).unwrap(),
        "in progress\n"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_staging_directory_without_an_owner_is_refused_and_kept() {
    let dir = scratch("no-owner");
    let staging = dir.join(".pose-staging");
    fs::create_dir_all(&staging).unwrap();
    assert!(matches!(
        FrameDump::create(&dir, &joints()),
        Err(PoseError::Locked { .. })
    ));
    assert!(staging.is_dir());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_at_a_frame_name_stops_create_before_any_step_and_keeps_the_commit() {
    let dir = scratch("svg-dir-preflight");
    write_run(&dir, 3).unwrap();
    let before = committed_bytes(&dir);
    let svg = dir.join(pose::SVG_DIR).join(pose::svg_file_name(2));
    let svg_bytes = fs::read(&svg).unwrap();
    fs::remove_file(&svg).unwrap();
    fs::create_dir(&svg).unwrap();
    assert!(FrameDump::create(&dir, &joints()).is_err());
    fs::remove_dir(&svg).unwrap();
    fs::write(&svg, svg_bytes).unwrap();
    assert_eq!(committed_bytes(&dir), before);
    assert!(!dir.join(".pose-staging").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_in_a_published_svg_slot_fails_finish_and_keeps_the_committed_run() {
    let dir = scratch("svg-publish-preflight");
    write_run(&dir, 3).unwrap();
    let before = committed_bytes(&dir);
    let mut dump = FrameDump::create(&dir, &joints()).unwrap();
    for step in 1..=3 {
        dump.push(&frame_at(step)).unwrap();
    }
    let svg = dir.join(pose::SVG_DIR).join(pose::svg_file_name(3));
    let svg_bytes = fs::read(&svg).unwrap();
    fs::remove_file(&svg).unwrap();
    fs::create_dir(&svg).unwrap();
    assert!(dump.finish().is_err());
    fs::remove_dir(&svg).unwrap();
    fs::write(&svg, svg_bytes).unwrap();
    assert!(!dir.join(".pose-staging").exists());
    assert_eq!(committed_bytes(&dir), before);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_at_frames_csv_stops_create_and_keeps_the_commit() {
    let dir = scratch("csv-dir-preflight");
    write_run(&dir, 3).unwrap();
    let before = committed_bytes(&dir);
    let csv = dir.join(pose::CSV_FILE);
    fs::remove_file(&csv).unwrap();
    fs::create_dir(&csv).unwrap();
    assert!(FrameDump::create(&dir, &joints()).is_err());
    fs::remove_dir(&csv).unwrap();
    fs::write(&csv, &before[0].1).unwrap();
    assert_eq!(committed_bytes(&dir), before);
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn an_unwritable_svg_directory_fails_create_and_keeps_the_commit() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("readonly-svg");
    write_run(&dir, 3).unwrap();
    let before = committed_bytes(&dir);
    let svg_dir = dir.join(pose::SVG_DIR);
    fs::set_permissions(&svg_dir, fs::Permissions::from_mode(0o555)).unwrap();
    let probe = svg_dir.join(".write-check");
    let writable = fs::File::create(&probe).is_ok();
    let _ = fs::remove_file(&probe);
    let result = FrameDump::create(&dir, &joints());
    fs::set_permissions(&svg_dir, fs::Permissions::from_mode(0o755)).unwrap();
    if writable {
        // Running as root: permission bits do not restrict writes, so there is
        // nothing to refuse. The committed run must still be untouched.
        drop(result);
    } else {
        assert!(result.is_err());
    }
    assert_eq!(committed_bytes(&dir), before);
    assert!(!dir.join(".pose-staging").exists());
    let _ = fs::remove_dir_all(&dir);
}
