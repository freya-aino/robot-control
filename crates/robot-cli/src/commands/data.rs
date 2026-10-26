//! `robot data inspect <path>` — PLAN.md §4's first proposed command.
//!
//! Inspects a file this workspace might have produced: the bounded TOML
//! subset of `robot_artifacts::toml`, or the binary chunk format of
//! `robot_artifacts::chunk`. Detection is by trying each in turn, not by
//! file extension, since this project's own artifact formats carry their
//! own framing (a magic number for chunks) rather than relying on a name.

use super::EXIT_FAILURE;
use robot_artifacts::chunk::{Chunk, ChunkLimits};
use robot_artifacts::toml;
use std::path::Path;

/// Runs `robot data inspect <path>`. Returns the process exit code.
pub fn inspect(path: &Path) -> i32 {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: could not read {}: {e}", path.display());
            return EXIT_FAILURE;
        }
    };

    if let Ok(chunk) = Chunk::decode(&bytes, None, ChunkLimits::default()) {
        print_chunk(path, &chunk);
        return 0;
    }

    match std::str::from_utf8(&bytes) {
        Ok(text) => match toml::parse(text) {
            Ok(table) => {
                print_table(path, &table);
                0
            }
            Err(e) => {
                eprintln!(
                    "error: {} is neither a recognized chunk (bad magic) nor valid TOML subset: {e}",
                    path.display()
                );
                EXIT_FAILURE
            }
        },
        Err(_) => {
            eprintln!(
                "error: {} is not a recognized chunk and is not valid UTF-8 text",
                path.display()
            );
            EXIT_FAILURE
        }
    }
}

fn print_chunk(path: &Path, chunk: &Chunk) {
    println!("{}: binary chunk", path.display());
    println!("  rows:            {}", chunk.rows);
    println!("  cols:            {}", chunk.cols);
    println!("  layout:          {:?}", chunk.layout);
    println!("  dtype:           {:?}", chunk.dtype);
    println!("  schema_digest:   {}", chunk.schema_digest);
    println!(
        "  values (first 8 of {}): {:?}",
        chunk.values.len(),
        &chunk.values[..chunk.values.len().min(8)]
    );
}

fn print_table(path: &Path, table: &toml::Table) {
    println!("{}: TOML-subset document", path.display());
    for (key, value) in table.iter() {
        println!("  {key} = {value:?}");
    }
    if table.is_empty() {
        println!("  (empty table)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("robot-cli-test-{name}-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn inspecting_a_toml_subset_document_succeeds() {
        let path = write_temp("toml", b"name = \"arm\"\nversion = 1\n");
        assert_eq!(inspect(&path), 0);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn inspecting_a_chunk_succeeds() {
        use robot_artifacts::chunk::{DType, Layout};
        use robot_artifacts::hash::Digest;
        let digest = Digest::from_hex(&"0".repeat(64)).unwrap();
        let chunk = Chunk::new(
            2,
            2,
            Layout::ArrayOfStructures,
            DType::F64,
            digest,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let path = write_temp("chunk", &chunk.encode());
        assert_eq!(inspect(&path), 0);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn inspecting_garbage_fails_clearly() {
        let path = write_temp("garbage", &[0xFFu8; 4]);
        assert_eq!(inspect(&path), EXIT_FAILURE);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn inspecting_a_missing_file_fails_clearly() {
        let path = std::env::temp_dir().join("robot-cli-test-definitely-missing-file");
        assert_eq!(inspect(&path), EXIT_FAILURE);
    }
}
