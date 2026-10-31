//! Composes a USD stage and prints its prim tree with types and selected
//! attributes.
//!
//! Usage: `cargo run -p robot-usd --example usd_dump -- <root layer> [prim path] [depth]`
//!
//! With no prim path the whole tree is printed to `depth` levels (default 3).
//! With a prim path only that prim's subtree is printed, to any depth.

use robot_usd::{Prim, Stage, Value};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(root) = args.first() else {
        eprintln!("usage: usd_dump <root layer> [prim path] [depth]");
        std::process::exit(2);
    };
    let stage = match Stage::compose(Path::new(root)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("compose failed: {e}");
            std::process::exit(1);
        }
    };
    println!("layers: {}", stage.layers.join(", "));
    let (start, depth): (Vec<&Prim>, usize) = match args.get(1) {
        Some(path) if path.starts_with('/') => {
            let depth = args
                .get(2)
                .and_then(|d| d.parse().ok())
                .unwrap_or(usize::MAX);
            match stage.prim(path) {
                Some(p) => (vec![p], depth),
                None => {
                    eprintln!("no prim at {path}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(3);
            (stage.prims.iter().collect(), depth)
        }
    };
    for p in start {
        print_prim(p, 0, depth);
    }
}

fn print_prim(p: &Prim, indent: usize, depth: usize) {
    let pad = "  ".repeat(indent);
    let ty = p.type_name.as_deref().unwrap_or("-");
    println!("{pad}{} [{ty}] {}", p.name, p.path);
    for prop in &p.properties {
        println!("{pad}  .{} {}", prop.name, summarize(prop));
    }
    if depth > 0 {
        for c in &p.children {
            print_prim(c, indent + 1, depth - 1);
        }
    } else if !p.children.is_empty() {
        println!("{pad}  ... {} children", p.children.len());
    }
}

fn summarize(prop: &robot_usd::Property) -> String {
    let mut parts = Vec::new();
    if let Some(t) = &prop.type_name {
        parts.push(t.clone());
    }
    if let Some(d) = &prop.default {
        parts.push(format!("= {}", short(d)));
    }
    if !prop.targets.is_empty() {
        parts.push(format!("targets {:?}", prop.targets));
    }
    if let Some(ts) = &prop.time_samples {
        parts.push(format!("{} time samples", ts.len()));
    }
    parts.join(" ")
}

fn short(v: &Value) -> String {
    let s = format!("{v:?}");
    if s.len() > 96 {
        format!("{}...", &s[..96])
    } else {
        s
    }
}
