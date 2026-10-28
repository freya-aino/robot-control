//! Checkpoint and resume for [`PpoAgent`] (SPECS.md Section 9.2; ADR 0007).
//!
//! A checkpoint is a directory holding one `<name>.chunk` file per piece of
//! training state and a `manifest.toml` written last. The manifest names each
//! chunk's content identity, so a load rejects any file that was truncated,
//! corrupted, swapped, or written under another schema before any of it is
//! used. Loading is all-or-nothing: every chunk is read and validated into
//! locals, and the agent is constructed only once all of them have passed.
//!
//! Every `f64` is stored as a `DType::F64` value, so the bits survive the
//! round trip exactly. Counters and RNG words are stored as `DType::U8` bytes.

use std::fmt;
use std::fs;
use std::path::Path;

use crate::backprop::Gradients;
use crate::gaussian_policy::GaussianPolicy;
use crate::optimizer::{AdamConfig, AdamState};
use crate::ppo::{PpoAgent, PpoConfig};
use robot_artifacts::chunk::{Chunk, ChunkLimits, DType, Layout};
use robot_artifacts::hash::{digest_parts, ContentId, Digest};
use robot_artifacts::store::write_atomic;
use robot_artifacts::toml::{self, Table, TomlError, TomlLimits, Value};
use robot_nn::layer::{Activation, Linear};
use robot_nn::mlp::{Mlp, ParamGroup};
use robot_types::obs::Normalizer;
use robot_types::rng::{Rng, RngSnapshot, SeedStream};
use robot_types::tier::Generation;

const SCHEMA: &str = "robot-learn.checkpoint";
const SCHEMA_VERSION: i64 = 2;
const SCHEMA_TAG: &[u8] = b"robot-learn.checkpoint.v2";
const MANIFEST: &str = "manifest.toml";

/// Floats in the `config` chunk: six PPO scalars, then two Adam configs of four.
const CONFIG_LEN: usize = 14;
/// u64 words in the `counters` chunk.
const COUNTERS_LEN: usize = 6;
/// u64 words in the `rng` chunk: state, increment, seed, stream.
const RNG_LEN: usize = 4;

/// Every chunk a checkpoint holds, with its element type. A future schema
/// adds an entry here and bumps [`SCHEMA_VERSION`].
const CHUNKS: [(&str, DType); 10] = [
    ("policy", DType::F64),
    ("policy-log-std", DType::F64),
    ("value", DType::F64),
    ("policy-adam", DType::F64),
    ("value-adam", DType::F64),
    ("normalizer", DType::F64),
    ("config", DType::F64),
    ("counters", DType::U8),
    ("rng", DType::U8),
    ("param-groups", DType::U8),
];

/// Why a checkpoint could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// A file could not be read or written.
    Io {
        /// The path involved.
        path: String,
        /// What went wrong.
        message: String,
    },
    /// The manifest is malformed or names an unexpected schema.
    Manifest(String),
    /// A chunk is missing, corrupt, or written under another schema.
    Chunk {
        /// The chunk name.
        name: String,
        /// What went wrong.
        message: String,
    },
    /// The chunks are individually valid but do not describe a consistent agent.
    Invalid(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "checkpoint I/O at {path}: {message}"),
            Self::Manifest(message) => write!(f, "checkpoint manifest: {message}"),
            Self::Chunk { name, message } => write!(f, "checkpoint chunk {name}: {message}"),
            Self::Invalid(message) => write!(f, "invalid checkpoint: {message}"),
        }
    }
}

impl core::error::Error for CheckpointError {}

/// Provenance recorded in a checkpoint's manifest alongside its state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointMetadata {
    /// The source revision the checkpoint was saved under, or `"unknown"`
    /// when the writer did not supply one.
    pub source_revision: String,
}

/// Writes `agent` and `rng` as a new checkpoint in `dir`.
///
/// Refuses to write over an existing checkpoint: the chunks are written before
/// the manifest, so overwriting would leave a window in which the old manifest
/// names new chunk bytes. Each checkpoint goes to a fresh directory.
pub fn save(dir: &Path, agent: &PpoAgent, rng: &Rng) -> Result<(), CheckpointError> {
    save_with_metadata(dir, agent, rng, None)
}

/// Writes a checkpoint with an optional repository source revision.
///
/// When `source_revision` is `None`, the manifest records `"unknown"`. The
/// convenience [`save`] function uses this default because an agent does not
/// own the revision of the program that trained it.
pub fn save_with_metadata(
    dir: &Path,
    agent: &PpoAgent,
    rng: &Rng,
    source_revision: Option<&str>,
) -> Result<(), CheckpointError> {
    if dir.join(MANIFEST).exists() {
        return Err(CheckpointError::Io {
            path: dir.display().to_string(),
            message: "a checkpoint already exists here; use a fresh directory".to_owned(),
        });
    }
    let source_revision = source_revision.unwrap_or("unknown");
    if source_revision.is_empty() {
        return Err(invalid("source revision must not be empty"));
    }
    agent
        .policy
        .frozen_mask(&agent.policy_groups)
        .map_err(|e| invalid(format!("policy parameter groups: {e}")))?;
    agent
        .value_net
        .frozen_mask(&agent.value_groups)
        .map_err(|e| invalid(format!("value parameter groups: {e}")))?;

    let mut digests = Table::new();
    for (name, dtype, values) in payloads(agent, rng) {
        let chunk = make_chunk(name, dtype, values)?;
        let bytes = chunk.encode();
        write_atomic(&dir.join(chunk_file(name)), &bytes).map_err(|e| io_error(dir, &e))?;
        digests.insert(name, Value::String(ContentId::of(&bytes).to_string()));
    }

    let (policy_widths, policy_activations) = topology(&agent.policy.mean_net);
    let (value_widths, value_activations) = topology(&agent.value_net);
    let mut manifest = Table::new();
    manifest.insert("schema", Value::String(SCHEMA.to_owned()));
    manifest.insert("schema_version", Value::Integer(SCHEMA_VERSION));
    manifest.insert("source_revision", Value::String(source_revision.to_owned()));
    manifest.insert("seed", Value::String(rng.origin().seed.to_string()));
    manifest.insert("obs_width", Value::Integer(agent.normalizer.width() as i64));
    manifest.insert("policy_widths", widths_value(&policy_widths));
    manifest.insert("policy_activations", activations_value(&policy_activations));
    manifest.insert("value_widths", widths_value(&value_widths));
    manifest.insert("value_activations", activations_value(&value_activations));
    manifest.insert("chunks", Value::Table(digests));

    write_atomic(&dir.join(MANIFEST), toml::to_string(&manifest).as_bytes())
        .map_err(|e| io_error(dir, &e))
}

/// Reads a checkpoint written by [`save`], returning the agent and RNG exactly
/// as they were saved.
///
/// Any failure returns an error and no agent: nothing is constructed until
/// every chunk has been verified against the manifest and decoded.
pub fn load(dir: &Path) -> Result<(PpoAgent, Rng), CheckpointError> {
    load_with_metadata(dir).map(|(agent, rng, _)| (agent, rng))
}

/// Reads a checkpoint like [`load`], and also returns its recorded
/// [`CheckpointMetadata`]. Checkpoints written before the source revision
/// was recorded report `"unknown"`.
pub fn load_with_metadata(
    dir: &Path,
) -> Result<(PpoAgent, Rng, CheckpointMetadata), CheckpointError> {
    let manifest_bytes = read_file(&dir.join(MANIFEST))?;
    let text = String::from_utf8(manifest_bytes)
        .map_err(|_| CheckpointError::Manifest("manifest is not UTF-8".to_owned()))?;
    let manifest = toml::parse_with_limits(&text, TomlLimits::default()).map_err(manifest_error)?;
    check_header(&manifest)?;
    let source_revision = match manifest.get("source_revision") {
        Some(_) => manifest
            .require_str("source_revision")
            .map_err(manifest_error)?
            .to_owned(),
        None => "unknown".to_owned(),
    };
    if source_revision.is_empty() {
        return Err(CheckpointError::Manifest(
            "source_revision must not be empty".to_owned(),
        ));
    }

    let digests = manifest.require_table("chunks").map_err(manifest_error)?;
    let names_match =
        digests.len() == CHUNKS.len() && CHUNKS.iter().all(|(name, _)| digests.get(name).is_some());
    if !names_match {
        return Err(CheckpointError::Manifest(
            "chunk table does not name exactly the expected chunks".to_owned(),
        ));
    }

    let obs_width = manifest_usize(&manifest, "obs_width")?;
    let policy_widths = read_widths(&manifest, "policy_widths")?;
    let policy_activations = read_activations(&manifest, "policy_activations")?;
    let value_widths = read_widths(&manifest, "value_widths")?;
    let value_activations = read_activations(&manifest, "value_activations")?;
    let manifest_seed = manifest
        .require_str("seed")
        .map_err(manifest_error)?
        .parse::<u64>()
        .map_err(|_| CheckpointError::Manifest("seed is not a u64".to_owned()))?;

    let policy_mlp = read_network(
        &read_chunk(dir, digests, "policy", DType::F64)?,
        &policy_widths,
        &policy_activations,
    )?;

    if policy_mlp.input_width() != obs_width {
        return Err(invalid(
            "policy input width does not match observation width",
        ));
    }

    let log_std = read_chunk(dir, digests, "policy-log-std", DType::F64)?;
    let policy = GaussianPolicy::from_parts(policy_mlp, log_std)
        .ok_or_else(|| invalid("log-std length does not match the policy output width"))?;
    let value_net = read_network(
        &read_chunk(dir, digests, "value", DType::F64)?,
        &value_widths,
        &value_activations,
    )?;

    if value_net.input_width() != obs_width {
        return Err(invalid(
            "value input width does not match observation width",
        ));
    }
    if value_net.output_width() != 1 {
        return Err(invalid("value output width is not 1"));
    }

    let counters = read_u64s(
        &read_chunk(dir, digests, "counters", DType::U8)?,
        COUNTERS_LEN,
    )?;
    let rng_words = read_u64s(&read_chunk(dir, digests, "rng", DType::U8)?, RNG_LEN)?;
    let config = read_config(
        &read_chunk(dir, digests, "config", DType::F64)?,
        counters[4],
    )?;
    let policy_adam = read_adam(
        &read_chunk(dir, digests, "policy-adam", DType::F64)?,
        &policy.mean_net,
        counters[1],
    )?;
    let value_adam = read_adam(
        &read_chunk(dir, digests, "value-adam", DType::F64)?,
        &value_net,
        counters[2],
    )?;
    let frozen = match counters[5] {
        0 => false,
        1 => true,
        _ => return Err(invalid("normalizer frozen flag is not 0 or 1")),
    };
    let normalizer = read_normalizer(
        &read_chunk(dir, digests, "normalizer", DType::F64)?,
        obs_width,
        counters[3],
        frozen,
    )?;

    let (policy_groups, value_groups) =
        read_groups(&read_chunk(dir, digests, "param-groups", DType::U8)?)?;
    policy
        .frozen_mask(&policy_groups)
        .map_err(|e| invalid(format!("policy parameter groups: {e}")))?;
    value_net
        .frozen_mask(&value_groups)
        .map_err(|e| invalid(format!("value parameter groups: {e}")))?;

    let snapshot = RngSnapshot {
        state: rng_words[0],
        increment: rng_words[1],
        origin: SeedStream::new(rng_words[2], rng_words[3]),
    };
    let rng = Rng::restore(snapshot).map_err(invalid)?;
    if rng.origin().seed != manifest_seed {
        return Err(invalid("manifest seed does not match the saved RNG stream"));
    }

    let agent = PpoAgent {
        policy,
        value_net,
        policy_adam,
        value_adam,
        normalizer,
        config,
        generation: Generation(counters[0]),
        policy_groups,
        value_groups,
    };
    Ok((agent, rng, CheckpointMetadata { source_revision }))
}

/// Every chunk's values, in the order of [`CHUNKS`].
fn payloads(agent: &PpoAgent, rng: &Rng) -> Vec<(&'static str, DType, Vec<f64>)> {
    let config = &agent.config;
    let mut config_values = vec![
        config.gamma,
        config.lambda,
        config.clip_ratio,
        config.value_loss_weight,
        config.entropy_weight,
        config.log_std_learning_rate,
    ];
    push_adam_config(&mut config_values, &config.policy_adam);
    push_adam_config(&mut config_values, &config.value_adam);

    let mut counters = Vec::with_capacity(COUNTERS_LEN * 8);
    push_u64s(
        &mut counters,
        &[
            agent.generation.0,
            agent.policy_adam.step(),
            agent.value_adam.step(),
            agent.normalizer.count(),
            config.epochs as u64,
            u64::from(agent.normalizer.is_frozen()),
        ],
    );

    let snapshot = rng.snapshot();
    let mut rng_bytes = Vec::with_capacity(RNG_LEN * 8);
    push_u64s(
        &mut rng_bytes,
        &[
            snapshot.state,
            snapshot.increment,
            snapshot.origin.seed,
            snapshot.origin.stream,
        ],
    );

    let mut normalizer = Vec::new();
    normalizer.extend_from_slice(agent.normalizer.mean());
    normalizer.extend_from_slice(agent.normalizer.m2());
    normalizer.push(agent.normalizer.clip());
    normalizer.push(agent.normalizer.epsilon());

    let mut policy_adam = Vec::new();
    push_gradients(&mut policy_adam, agent.policy_adam.first_moment());
    push_gradients(&mut policy_adam, agent.policy_adam.second_moment());
    let mut value_adam = Vec::new();
    push_gradients(&mut value_adam, agent.value_adam.first_moment());
    push_gradients(&mut value_adam, agent.value_adam.second_moment());

    vec![
        ("policy", DType::F64, network_values(&agent.policy.mean_net)),
        ("policy-log-std", DType::F64, agent.policy.log_std.clone()),
        ("value", DType::F64, network_values(&agent.value_net)),
        ("policy-adam", DType::F64, policy_adam),
        ("value-adam", DType::F64, value_adam),
        ("normalizer", DType::F64, normalizer),
        ("config", DType::F64, config_values),
        ("counters", DType::U8, counters),
        ("rng", DType::U8, rng_bytes),
        ("param-groups", DType::U8, encode_groups(agent)),
    ]
}

/// The parameter-group tables: a policy list, then a value list. Each list is
/// a count, then per group a name (length, then bytes), the range start and
/// end, and the frozen flag, all as little-endian u64 words stored byte-wise.
fn encode_groups(agent: &PpoAgent) -> Vec<f64> {
    let mut out = Vec::new();
    push_group_list(&mut out, &agent.policy_groups);
    push_group_list(&mut out, &agent.value_groups);
    out
}

fn push_group_list(out: &mut Vec<f64>, groups: &[ParamGroup]) {
    push_u64s(out, &[groups.len() as u64]);
    for group in groups {
        push_u64s(out, &[group.name.len() as u64]);
        out.extend(group.name.bytes().map(f64::from));
        push_u64s(
            out,
            &[
                group.range.start as u64,
                group.range.end as u64,
                u64::from(group.frozen),
            ],
        );
    }
}

fn read_groups(values: &[f64]) -> Result<(Vec<ParamGroup>, Vec<ParamGroup>), CheckpointError> {
    let bytes = values
        .iter()
        .map(|value| {
            if (0.0..=255.0).contains(value) && value.fract() == 0.0 {
                Ok(*value as u8)
            } else {
                Err(invalid("param-groups chunk holds a non-byte value"))
            }
        })
        .collect::<Result<Vec<u8>, _>>()?;
    let mut reader = GroupReader {
        bytes: &bytes,
        at: 0,
    };
    let policy = read_group_list(&mut reader)?;
    let value = read_group_list(&mut reader)?;
    if reader.at != bytes.len() {
        return Err(invalid("param-groups chunk has trailing bytes"));
    }
    Ok((policy, value))
}

struct GroupReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> GroupReader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| invalid("param-groups chunk is truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn word(&mut self) -> Result<u64, CheckpointError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }
}

fn read_group_list(reader: &mut GroupReader<'_>) -> Result<Vec<ParamGroup>, CheckpointError> {
    let count = reader.word()?;
    let mut groups = Vec::new();
    for _ in 0..count {
        let len = usize::try_from(reader.word()?)
            .map_err(|_| invalid("parameter group name is too long"))?;
        let name = std::str::from_utf8(reader.take(len)?)
            .map_err(|_| invalid("parameter group name is not UTF-8"))?
            .to_owned();
        let start = usize::try_from(reader.word()?)
            .map_err(|_| invalid("parameter group range does not fit usize"))?;
        let end = usize::try_from(reader.word()?)
            .map_err(|_| invalid("parameter group range does not fit usize"))?;
        let frozen = match reader.word()? {
            0 => false,
            1 => true,
            _ => return Err(invalid("parameter group frozen flag is not 0 or 1")),
        };
        groups.push(ParamGroup {
            name,
            range: start..end,
            frozen,
        });
    }
    Ok(groups)
}

/// The schema digest a chunk is written and read under. Binding the chunk
/// name and element type means a chunk cannot be reinterpreted as another.
fn schema_digest(name: &str, dtype: DType) -> Digest {
    digest_parts(&[
        SCHEMA_TAG,
        name.as_bytes(),
        dtype.id().as_bytes(),
        Layout::StructureOfArrays.id().as_bytes(),
    ])
    .digest()
}

fn make_chunk(name: &str, dtype: DType, values: Vec<f64>) -> Result<Chunk, CheckpointError> {
    let rows =
        u32::try_from(values.len()).map_err(|_| invalid(format!("chunk {name} is too large")))?;
    Chunk::new(
        rows,
        1,
        Layout::StructureOfArrays,
        dtype,
        schema_digest(name, dtype),
        values,
    )
    .map_err(|e| chunk_error(name, e))
}

fn read_chunk(
    dir: &Path,
    digests: &Table,
    name: &str,
    dtype: DType,
) -> Result<Vec<f64>, CheckpointError> {
    let expected = digests.require_str(name).map_err(manifest_error)?;
    let expected = ContentId::parse(expected)
        .map_err(|e| CheckpointError::Manifest(format!("chunk {name}: {e}")))?;
    let bytes = read_file(&dir.join(chunk_file(name)))?;
    expected.verify(&bytes).map_err(|e| chunk_error(name, e))?;
    let chunk = Chunk::decode(
        &bytes,
        Some(schema_digest(name, dtype)),
        ChunkLimits::default(),
    )
    .map_err(|e| chunk_error(name, e))?;
    if chunk.dtype() != dtype {
        return Err(chunk_error(name, "element type does not match the schema"));
    }
    Ok(chunk.values().to_vec())
}

fn read_network(
    values: &[f64],
    widths: &[usize],
    activations: &[Activation],
) -> Result<Mlp, CheckpointError> {
    if activations.len() + 1 != widths.len() {
        return Err(invalid("activation count does not match layer count"));
    }
    let expected = widths
        .windows(2)
        .try_fold(0usize, |acc, w| {
            w[0].checked_mul(w[1])
                .and_then(|n| n.checked_add(w[1]))
                .and_then(|n| n.checked_add(acc))
        })
        .ok_or_else(|| invalid("network topology is too large"))?;
    if values.len() != expected {
        return Err(invalid(format!(
            "network holds {} values, topology needs {expected}",
            values.len()
        )));
    }
    let mut at = 0;
    let mut layers = Vec::with_capacity(widths.len() - 1);
    for pair in widths.windows(2) {
        let (inputs, outputs) = (pair[0], pair[1]);
        let weight = values[at..at + inputs * outputs].to_vec();
        at += inputs * outputs;
        let bias = values[at..at + outputs].to_vec();
        at += outputs;
        layers.push(
            Linear::new(inputs, outputs, weight, bias).map_err(|e| invalid(format!("{e:?}")))?,
        );
    }
    Mlp::new(layers, activations.to_vec()).map_err(|e| invalid(format!("{e:?}")))
}

fn network_values(mlp: &Mlp) -> Vec<f64> {
    let mut out = Vec::new();
    for layer in mlp.layers() {
        out.extend_from_slice(layer.weight());
        out.extend_from_slice(layer.bias());
    }
    out
}

fn topology(mlp: &Mlp) -> (Vec<usize>, Vec<Activation>) {
    let mut widths = vec![mlp.input_width()];
    widths.extend(mlp.layers().iter().map(Linear::out_features));
    (widths, mlp.activations().to_vec())
}

fn parameter_count(mlp: &Mlp) -> usize {
    mlp.layers()
        .iter()
        .map(|l| l.weight().len() + l.bias().len())
        .sum()
}

fn push_gradients(out: &mut Vec<f64>, grads: &Gradients) {
    for (weight, bias) in grads.weight.iter().zip(&grads.bias) {
        out.extend_from_slice(weight);
        out.extend_from_slice(bias);
    }
}

fn unflatten(values: &[f64], mlp: &Mlp) -> Gradients {
    let mut grads = Gradients::zeros_like(mlp);
    let mut at = 0;
    for (weight, bias) in grads.weight.iter_mut().zip(grads.bias.iter_mut()) {
        let (w_len, b_len) = (weight.len(), bias.len());
        weight.copy_from_slice(&values[at..at + w_len]);
        at += w_len;
        bias.copy_from_slice(&values[at..at + b_len]);
        at += b_len;
    }
    grads
}

fn read_adam(values: &[f64], mlp: &Mlp, step: u64) -> Result<AdamState, CheckpointError> {
    let count = parameter_count(mlp);
    if values.len() != 2 * count {
        return Err(invalid(format!(
            "Adam chunk holds {} values, expected {}",
            values.len(),
            2 * count
        )));
    }
    let m = unflatten(&values[..count], mlp);
    let v = unflatten(&values[count..], mlp);
    AdamState::from_parts(mlp, m, v, step).map_err(|e| CheckpointError::Invalid(e.to_owned()))
}

fn push_adam_config(out: &mut Vec<f64>, config: &AdamConfig) {
    out.push(config.learning_rate);
    out.push(config.beta1);
    out.push(config.beta2);
    out.push(config.epsilon);
}

fn read_config(values: &[f64], epochs: u64) -> Result<PpoConfig, CheckpointError> {
    if values.len() != CONFIG_LEN {
        return Err(invalid("config chunk has the wrong length"));
    }
    let adam = |at: usize| AdamConfig {
        learning_rate: values[at],
        beta1: values[at + 1],
        beta2: values[at + 2],
        epsilon: values[at + 3],
    };
    Ok(PpoConfig {
        gamma: values[0],
        lambda: values[1],
        clip_ratio: values[2],
        value_loss_weight: values[3],
        entropy_weight: values[4],
        log_std_learning_rate: values[5],
        policy_adam: adam(6),
        value_adam: adam(10),
        epochs: usize::try_from(epochs).map_err(|_| invalid("epoch count does not fit usize"))?,
    })
}

fn read_normalizer(
    values: &[f64],
    width: usize,
    count: u64,
    frozen: bool,
) -> Result<Normalizer, CheckpointError> {
    let expected = width
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| invalid("normalizer width is too large"))?;
    if values.len() != expected {
        return Err(invalid("normalizer chunk has the wrong length"));
    }
    Normalizer::from_state(
        values[..width].to_vec(),
        values[width..2 * width].to_vec(),
        count,
        values[2 * width],
        values[2 * width + 1],
        frozen,
    )
    .map_err(|e| invalid(format!("{e:?}")))
}

fn push_u64s(out: &mut Vec<f64>, words: &[u64]) {
    for word in words {
        for byte in word.to_le_bytes() {
            out.push(f64::from(byte));
        }
    }
}

fn read_u64s(values: &[f64], count: usize) -> Result<Vec<u64>, CheckpointError> {
    if values.len() != count * 8 {
        return Err(invalid("integer chunk has the wrong length"));
    }
    let mut words = Vec::with_capacity(count);
    for word in values.chunks_exact(8) {
        let mut bytes = [0u8; 8];
        for (slot, value) in bytes.iter_mut().zip(word) {
            if !(0.0..=255.0).contains(value) || value.fract() != 0.0 {
                return Err(invalid("integer chunk holds a non-byte value"));
            }
            *slot = *value as u8;
        }
        words.push(u64::from_le_bytes(bytes));
    }
    Ok(words)
}

fn check_header(manifest: &Table) -> Result<(), CheckpointError> {
    let schema = manifest.require_str("schema").map_err(manifest_error)?;
    if schema != SCHEMA {
        return Err(CheckpointError::Manifest(format!(
            "schema {schema:?} is not {SCHEMA:?}"
        )));
    }
    let version = manifest
        .require_integer("schema_version")
        .map_err(manifest_error)?;
    if version != SCHEMA_VERSION {
        return Err(CheckpointError::Manifest(format!(
            "schema version {version} is not supported (expected {SCHEMA_VERSION})"
        )));
    }
    Ok(())
}

fn manifest_usize(manifest: &Table, key: &str) -> Result<usize, CheckpointError> {
    let value = manifest.require_integer(key).map_err(manifest_error)?;
    usize::try_from(value)
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| CheckpointError::Manifest(format!("{key} must be a positive integer")))
}

fn read_widths(manifest: &Table, key: &str) -> Result<Vec<usize>, CheckpointError> {
    manifest
        .require_array(key)
        .map_err(manifest_error)?
        .iter()
        .map(|v| {
            v.as_integer()
                .and_then(|w| usize::try_from(w).ok())
                .filter(|w| *w > 0)
                .ok_or_else(|| CheckpointError::Manifest(format!("{key} holds a bad width")))
        })
        .collect()
}

fn read_activations(manifest: &Table, key: &str) -> Result<Vec<Activation>, CheckpointError> {
    manifest
        .require_array(key)
        .map_err(manifest_error)?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| CheckpointError::Manifest(format!("{key} holds a non-string")))
                .and_then(parse_activation)
        })
        .collect()
}

fn widths_value(widths: &[usize]) -> Value {
    Value::Array(widths.iter().map(|w| Value::Integer(*w as i64)).collect())
}

fn activations_value(activations: &[Activation]) -> Value {
    Value::Array(
        activations
            .iter()
            .map(|a| Value::String(activation_id(*a).to_owned()))
            .collect(),
    )
}

const fn activation_id(activation: Activation) -> &'static str {
    match activation {
        Activation::Identity => "identity",
        Activation::Tanh => "tanh",
        Activation::ReLU => "relu",
        Activation::Sigmoid => "sigmoid",
    }
}

fn parse_activation(text: &str) -> Result<Activation, CheckpointError> {
    match text {
        "identity" => Ok(Activation::Identity),
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::ReLU),
        "sigmoid" => Ok(Activation::Sigmoid),
        other => Err(CheckpointError::Manifest(format!(
            "unknown activation {other:?}"
        ))),
    }
}

fn chunk_file(name: &str) -> String {
    format!("{name}.chunk")
}

fn read_file(path: &Path) -> Result<Vec<u8>, CheckpointError> {
    fs::read(path).map_err(|e| CheckpointError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

fn io_error(dir: &Path, error: &dyn fmt::Display) -> CheckpointError {
    CheckpointError::Io {
        path: dir.display().to_string(),
        message: error.to_string(),
    }
}

fn manifest_error(error: TomlError) -> CheckpointError {
    CheckpointError::Manifest(error.to_string())
}

fn chunk_error(name: &str, error: impl fmt::Display) -> CheckpointError {
    CheckpointError::Chunk {
        name: name.to_owned(),
        message: error.to_string(),
    }
}

fn invalid(message: impl Into<String>) -> CheckpointError {
    CheckpointError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ppo::RolloutContext;
    use crate::rollout::RolloutBuffer;
    use robot_nn::init::initialize;
    use robot_sim::env::{Environment, PhysicsStep};
    use robot_types::ids::EnvId;
    use robot_types::transition::{RewardTerm, RewardTerms, Termination};

    fn value_net() -> Mlp {
        let hidden = Linear::new(2, 4, vec![0.1; 8], vec![0.0; 4]).unwrap();
        let out = Linear::new(4, 1, vec![0.05; 4], vec![0.0]).unwrap();
        Mlp::new(
            vec![hidden, out],
            vec![Activation::Tanh, Activation::Identity],
        )
        .unwrap()
    }

    fn single_layer_net(input_width: usize, output_width: usize) -> Mlp {
        let layer = Linear::new(
            input_width,
            output_width,
            vec![0.1; input_width * output_width],
            vec![0.0; output_width],
        )
        .unwrap();
        Mlp::new(vec![layer], vec![Activation::Identity]).unwrap()
    }

    fn agent_with_networks(
        policy_input_width: usize,
        policy_output_width: usize,
        value_input_width: usize,
        value_output_width: usize,
        obs_width: usize,
    ) -> PpoAgent {
        let policy = GaussianPolicy::new(
            single_layer_net(policy_input_width, policy_output_width),
            -0.5,
        );
        PpoAgent::new(
            policy,
            single_layer_net(value_input_width, value_output_width),
            PpoConfig::default(),
            obs_width,
        )
    }

    fn agent() -> PpoAgent {
        let policy = GaussianPolicy::new(value_net(), -0.5);
        let mut agent = PpoAgent::new(policy, value_net(), PpoConfig::default(), 2);
        agent.normalizer.observe(&[1.0, -2.0]).unwrap();
        agent.normalizer.observe(&[0.5, 3.0]).unwrap();
        let mut grads = Gradients::zeros_like(&agent.value_net);
        grads.weight[0][0] = 1.0;
        grads.bias[1][0] = -0.25;
        let config = AdamConfig::default();
        agent
            .value_adam
            .step_with(&mut agent.value_net, &grads, &config);
        agent
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "robot-learn-checkpoint-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn round_trip_restores_weights_normalizer_adam_and_rng() {
        let dir = scratch_dir("round-trip");
        let agent = agent();
        let mut rng = Rng::new(SeedStream::new(7, 1));
        rng.next_u64();

        save(&dir, &agent, &rng).unwrap();
        let manifest_text = fs::read_to_string(dir.join(MANIFEST)).unwrap();
        assert!(manifest_text.contains("source_revision = \"unknown\""));
        let (loaded, mut loaded_rng) = load(&dir).unwrap();

        assert_eq!(loaded.policy.mean_net, agent.policy.mean_net);
        assert_eq!(loaded.policy.log_std, agent.policy.log_std);
        assert_eq!(loaded.value_net, agent.value_net);
        assert_eq!(loaded.normalizer, agent.normalizer);
        assert_eq!(loaded.generation, agent.generation);
        assert_eq!(loaded.policy_adam.step(), agent.policy_adam.step());
        assert_eq!(loaded.value_adam.step(), agent.value_adam.step());
        assert_eq!(
            loaded.value_adam.first_moment().weight,
            agent.value_adam.first_moment().weight
        );
        assert_eq!(
            loaded.value_adam.second_moment().bias,
            agent.value_adam.second_moment().bias
        );
        assert_eq!(loaded_rng, rng);
        let mut original_rng = rng.clone();
        assert_eq!(loaded_rng.next_u64(), original_rng.next_u64());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_chunk_is_rejected_without_an_agent() {
        let dir = scratch_dir("corrupt");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        let path = dir.join("policy.chunk");
        let mut bytes = fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        fs::write(&path, bytes).unwrap();

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Chunk { ref name, .. }) if name == "policy"
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_manifest_is_rejected() {
        let dir = scratch_dir("truncated");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        let path = dir.join(MANIFEST);
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, &text[..text.len() / 2]).unwrap();

        assert!(load(&dir).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parameter_groups_round_trip_and_invalid_tables_are_rejected() {
        let dir = scratch_dir("groups-round-trip");
        let mut good = agent();
        let groups = vec![ParamGroup {
            name: "hidden".to_owned(),
            range: 0..2,
            frozen: true,
        }];
        good.set_value_groups(groups.clone()).unwrap();
        save(&dir, &good, &Rng::new(SeedStream::new(3, 0))).unwrap();
        let (loaded, _) = load(&dir).unwrap();
        assert_eq!(loaded.value_groups(), groups.as_slice());
        assert!(loaded.policy_groups().is_empty());
        let _ = fs::remove_dir_all(&dir);

        let bad_dir = scratch_dir("groups-out-of-bounds");
        let mut bad = agent();
        bad.value_groups = vec![ParamGroup {
            name: "too-far".to_owned(),
            range: 0..10_000,
            frozen: true,
        }];
        assert!(matches!(
            save(&bad_dir, &bad, &Rng::new(SeedStream::new(3, 0))),
            Err(CheckpointError::Invalid(ref m)) if m.contains("value parameter groups")
        ));
        assert!(!bad_dir.join(MANIFEST).exists());
        let _ = fs::remove_dir_all(&bad_dir);
    }

    #[test]
    fn corrupt_parameter_group_chunk_is_rejected_on_load() {
        let dir = scratch_dir("groups-corrupt-chunk");
        let mut good = agent();
        good.set_value_groups(vec![ParamGroup {
            name: "hidden".to_owned(),
            range: 0..2,
            frozen: true,
        }])
        .unwrap();
        save(&dir, &good, &Rng::new(SeedStream::new(3, 0))).unwrap();
        let path = dir.join(chunk_file("param-groups"));
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&path, bytes).unwrap();

        assert!(load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    fn list_values(out: &mut Vec<f64>, groups: &[(&str, u64, u64, u64)]) {
        push_u64s(out, &[groups.len() as u64]);
        for (name, start, end, frozen) in groups {
            push_u64s(out, &[name.len() as u64]);
            out.extend(name.bytes().map(f64::from));
            push_u64s(out, &[*start, *end, *frozen]);
        }
    }

    fn group_values(policy: &[(&str, u64, u64, u64)], value: &[(&str, u64, u64, u64)]) -> Vec<f64> {
        let mut out = Vec::new();
        list_values(&mut out, policy);
        list_values(&mut out, value);
        out
    }

    #[test]
    fn well_formed_group_chunk_decodes() {
        let values = group_values(&[("p", 0, 4, 1)], &[("v", 1, 2, 0)]);
        let (policy, value) = read_groups(&values).unwrap();
        assert_eq!(
            policy,
            vec![ParamGroup {
                name: "p".to_owned(),
                range: 0..4,
                frozen: true
            }]
        );
        assert_eq!(
            value,
            vec![ParamGroup {
                name: "v".to_owned(),
                range: 1..2,
                frozen: false
            }]
        );
    }

    #[test]
    fn truncated_group_chunk_is_rejected() {
        let mut values = group_values(&[("p", 0, 4, 1)], &[]);
        values.pop();
        assert!(matches!(
            read_groups(&values),
            Err(CheckpointError::Invalid(ref m)) if m.contains("truncated")
        ));
    }

    #[test]
    fn oversized_length_prefix_is_rejected_without_panic() {
        let mut values = Vec::new();
        push_u64s(&mut values, &[1, u64::MAX]);
        assert!(matches!(
            read_groups(&values),
            Err(CheckpointError::Invalid(ref m)) if m.contains("truncated")
        ));
        let mut huge_count = Vec::new();
        push_u64s(&mut huge_count, &[u64::MAX]);
        assert!(read_groups(&huge_count).is_err());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut values = group_values(&[], &[]);
        values.push(0.0);
        assert!(matches!(
            read_groups(&values),
            Err(CheckpointError::Invalid(ref m)) if m.contains("trailing bytes")
        ));
    }

    #[test]
    fn non_byte_values_are_rejected() {
        for bad in [256.0, -1.0, 0.5, f64::NAN] {
            let mut values = group_values(&[], &[]);
            values[0] = bad;
            assert!(matches!(
                read_groups(&values),
                Err(CheckpointError::Invalid(ref m)) if m.contains("non-byte")
            ));
        }
    }

    #[test]
    fn non_utf8_group_name_is_rejected() {
        let mut values = Vec::new();
        push_u64s(&mut values, &[1, 1]);
        values.push(255.0);
        push_u64s(&mut values, &[0, 1, 0]);
        push_u64s(&mut values, &[0]);
        assert!(matches!(
            read_groups(&values),
            Err(CheckpointError::Invalid(ref m)) if m.contains("UTF-8")
        ));
    }

    #[test]
    fn frozen_flag_other_than_zero_or_one_is_rejected() {
        let values = group_values(&[("p", 0, 4, 2)], &[]);
        assert!(matches!(
            read_groups(&values),
            Err(CheckpointError::Invalid(ref m)) if m.contains("frozen flag")
        ));
    }

    #[test]
    fn overlapping_group_table_is_rejected_on_load() {
        let dir = scratch_dir("groups-overlap-on-load");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();

        let chunk_path = dir.join(chunk_file("param-groups"));
        let old_bytes = fs::read(&chunk_path).unwrap();
        let old_digest = ContentId::of(&old_bytes).to_string();
        let values = group_values(&[("a", 0, 4, 1), ("b", 2, 6, 0)], &[]);
        let new_bytes = make_chunk("param-groups", DType::U8, values)
            .unwrap()
            .encode();
        let new_digest = ContentId::of(&new_bytes).to_string();
        fs::write(&chunk_path, &new_bytes).unwrap();
        let manifest_path = dir.join(MANIFEST);
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        fs::write(&manifest_path, manifest.replace(&old_digest, &new_digest)).unwrap();

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Invalid(ref m)) if m.contains("policy parameter groups")
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn supplied_source_revision_is_written_and_empty_revision_is_rejected() {
        let dir = scratch_dir("source-revision");
        let rng = Rng::new(SeedStream::new(3, 0));
        save_with_metadata(&dir, &agent(), &rng, Some("abc123")).unwrap();
        let manifest = fs::read_to_string(dir.join(MANIFEST)).unwrap();
        assert!(manifest.contains("source_revision = \"abc123\""));
        let (_, _, metadata) = load_with_metadata(&dir).unwrap();
        assert_eq!(metadata.source_revision, "abc123");
        let _ = fs::remove_dir_all(&dir);

        let empty_dir = scratch_dir("empty-source-revision");
        assert!(matches!(
            save_with_metadata(&empty_dir, &agent(), &rng, Some("")),
            Err(CheckpointError::Invalid(_))
        ));
        assert!(!empty_dir.join(MANIFEST).exists());
        let _ = fs::remove_dir_all(&empty_dir);
    }

    #[test]
    fn older_manifest_without_source_revision_loads_with_unknown_provenance() {
        let dir = scratch_dir("legacy-source-revision");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        let path = dir.join(MANIFEST);
        let manifest = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            manifest.replace("source_revision = \"unknown\"\n", ""),
        )
        .unwrap();

        let (_, _, metadata) = load_with_metadata(&dir).unwrap();
        assert_eq!(metadata.source_revision, "unknown");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_input_width_must_match_manifest_observation_width() {
        let dir = scratch_dir("policy-observation-width");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        let path = dir.join(MANIFEST);
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("obs_width = 2"));
        fs::write(&path, text.replace("obs_width = 2", "obs_width = 3")).unwrap();

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Invalid(ref message))
                if message.contains("policy input width")
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn value_input_width_must_match_observation_width() {
        let dir = scratch_dir("value-observation-width");
        let malformed = agent_with_networks(2, 1, 3, 1, 2);
        save(&dir, &malformed, &Rng::new(SeedStream::new(3, 0))).unwrap();

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Invalid(ref message))
                if message.contains("value input width")
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn value_network_must_have_one_output() {
        let dir = scratch_dir("value-output-width");
        let malformed = agent_with_networks(2, 1, 2, 2, 2);
        save(&dir, &malformed, &Rng::new(SeedStream::new(3, 0))).unwrap();

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Invalid(ref message))
                if message.contains("value output width")
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_refuses_to_overwrite_an_existing_checkpoint() {
        let dir = scratch_dir("overwrite");
        let rng = Rng::new(SeedStream::new(3, 0));
        save(&dir, &agent(), &rng).unwrap();
        assert!(matches!(
            save(&dir, &agent(), &rng),
            Err(CheckpointError::Io { .. })
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    struct ResumeEnv {
        steps: u32,
    }

    impl Environment for ResumeEnv {
        fn observation_width(&self) -> usize {
            2
        }
        fn action_width(&self) -> usize {
            1
        }
        fn reset(&mut self, _seed: SeedStream, observation: &mut [f64]) {
            self.steps = 0;
            observation.copy_from_slice(&[0.0, 1.0]);
        }
        fn step(&mut self, action: &[f64], _dt: robot_types::time::Duration) -> PhysicsStep {
            self.steps += 1;
            let t = f64::from(self.steps);
            PhysicsStep {
                observation: vec![0.25 * t - action[0], 1.0 - 0.1 * t],
                physics_failure: false,
            }
        }
    }

    fn resume_agent() -> PpoAgent {
        let mut rng = Rng::new(SeedStream::new(11, 1));
        let policy_net = initialize(
            &[2, 8, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap();
        let value_net = initialize(
            &[2, 8, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap();
        let config = PpoConfig {
            epochs: 2,
            ..PpoConfig::default()
        };
        PpoAgent::new(GaussianPolicy::new(policy_net, -0.5), value_net, config, 2)
    }

    fn run_updates(agent: &mut PpoAgent, rng: &mut Rng, updates: usize) {
        let task = |obs: &[f64], action: &[f64], _next: &[f64], step: u32, _fail: bool| {
            let mut terms = RewardTerms::new();
            terms
                .push(RewardTerm {
                    name: "track".to_owned(),
                    value: -(action[0] - 0.3).powi(2) - 0.1 * obs[0].abs(),
                    weight: 1.0,
                    is_style_term: false,
                })
                .unwrap();
            let termination = if step >= 5 {
                Termination::TimeLimit
            } else {
                Termination::Continuing
            };
            (terms, termination)
        };
        let schema = agent.transition_schema("resume-fixture/1.0");
        let mut buffer = RolloutBuffer::new(schema, 64);
        let mut env = ResumeEnv { steps: 0 };
        for _ in 0..updates {
            buffer.clear();
            agent.collect_rollout(
                &mut env,
                &task,
                &mut buffer,
                32,
                RolloutContext {
                    env_id: EnvId::new(0),
                    robot_profile: "resume-fixture/1.0",
                    episode_seed_base: 4,
                },
                rng,
            );
            agent.update(&buffer);
        }
    }

    fn push_bits(out: &mut Vec<u64>, values: &[f64]) {
        out.extend(values.iter().map(|v| v.to_bits()));
    }

    /// Every piece of state a resumed run depends on, as exact bit patterns:
    /// both networks, the policy log-std, both Adam moment sets and step
    /// counts, the normalizer statistics, and the next RNG draws.
    fn fingerprint(agent: &PpoAgent, rng: &Rng) -> Vec<u64> {
        let mut bits = Vec::new();
        push_bits(&mut bits, &network_values(&agent.policy.mean_net));
        push_bits(&mut bits, &agent.policy.log_std);
        push_bits(&mut bits, &network_values(&agent.value_net));
        for adam in [&agent.policy_adam, &agent.value_adam] {
            let mut moments = Vec::new();
            push_gradients(&mut moments, adam.first_moment());
            push_gradients(&mut moments, adam.second_moment());
            push_bits(&mut bits, &moments);
            bits.push(adam.step());
        }
        push_bits(&mut bits, agent.normalizer.mean());
        push_bits(&mut bits, agent.normalizer.m2());
        push_bits(
            &mut bits,
            &[agent.normalizer.clip(), agent.normalizer.epsilon()],
        );
        let mut stream = rng.clone();
        bits.extend((0..4).map(|_| stream.next_u64()));
        bits
    }

    fn assert_same_state(a: &PpoAgent, a_rng: &Rng, b: &PpoAgent, b_rng: &Rng) {
        assert_eq!(fingerprint(a, a_rng), fingerprint(b, b_rng));
        assert_eq!(a.policy.mean_net, b.policy.mean_net);
        assert_eq!(a.value_net, b.value_net);
        assert_eq!(a.normalizer, b.normalizer);
        assert_eq!(a.generation, b.generation);
        assert_eq!(a.policy_adam.step(), b.policy_adam.step());
        assert_eq!(a.value_adam.step(), b.value_adam.step());
        assert_eq!(format!("{:?}", a.config), format!("{:?}", b.config));
        assert_eq!(a_rng, b_rng);
    }

    #[test]
    fn resumed_training_matches_an_uninterrupted_run_bit_for_bit() {
        let dir = scratch_dir("resume-equivalence");
        let mut uninterrupted = resume_agent();
        let mut uninterrupted_rng = Rng::new(SeedStream::new(9, 2));
        run_updates(&mut uninterrupted, &mut uninterrupted_rng, 6);
        assert_ne!(
            network_values(&uninterrupted.policy.mean_net),
            network_values(&resume_agent().policy.mean_net),
            "training must change the policy, or the comparison proves nothing"
        );

        let mut resumed = resume_agent();
        let mut resumed_rng = Rng::new(SeedStream::new(9, 2));
        run_updates(&mut resumed, &mut resumed_rng, 3);
        save(&dir, &resumed, &resumed_rng).unwrap();
        let (mut restored, mut restored_rng) = load(&dir).unwrap();
        assert_same_state(&resumed, &resumed_rng, &restored, &restored_rng);

        run_updates(&mut restored, &mut restored_rng, 3);
        assert_same_state(&uninterrupted, &uninterrupted_rng, &restored, &restored_rng);

        let _ = fs::remove_dir_all(&dir);
    }

    const CHILD_CHECKPOINT: &str = "ROBOT_LEARN_CHILD_CHECKPOINT";
    const CHILD_OUTPUT: &str = "ROBOT_LEARN_CHILD_OUTPUT";

    #[test]
    fn checkpoint_is_loaded_by_a_separate_process() {
        let dir = scratch_dir("cross-process");
        let out = scratch_dir("cross-process-output");
        let mut agent = resume_agent();
        let mut rng = Rng::new(SeedStream::new(9, 2));
        run_updates(&mut agent, &mut rng, 2);
        save(&dir, &agent, &rng).unwrap();
        let expected = fingerprint(&agent, &rng);

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "checkpoint::tests::child_process_reads_checkpoint",
                "--test-threads=1",
            ])
            .env(CHILD_CHECKPOINT, &dir)
            .env(CHILD_OUTPUT, &out)
            .status()
            .unwrap();
        assert!(status.success(), "child process failed: {status}");

        let child: Vec<u64> = fs::read_to_string(&out)
            .unwrap()
            .split_whitespace()
            .map(|word| word.parse().unwrap())
            .collect();
        assert_eq!(child, expected);

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_file(&out);
    }

    /// Runs only inside the child process spawned by
    /// `checkpoint_is_loaded_by_a_separate_process`; a plain test run skips it.
    #[test]
    fn child_process_reads_checkpoint() {
        let (dir, out) = match (
            std::env::var_os(CHILD_CHECKPOINT),
            std::env::var_os(CHILD_OUTPUT),
        ) {
            (Some(dir), Some(out)) => (dir, out),
            _ => return,
        };
        let (agent, rng) = load(Path::new(&dir)).unwrap();
        let words: Vec<String> = fingerprint(&agent, &rng)
            .iter()
            .map(u64::to_string)
            .collect();
        fs::write(out, words.join(" ")).unwrap();
    }

    fn edit_manifest(dir: &Path, edit: impl FnOnce(&mut Table)) {
        let path = dir.join(MANIFEST);
        let text = fs::read_to_string(&path).unwrap();
        let mut manifest = toml::parse_with_limits(&text, TomlLimits::default()).unwrap();
        edit(&mut manifest);
        fs::write(&path, toml::to_string(&manifest)).unwrap();
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let dir = scratch_dir("schema-version");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        edit_manifest(&dir, |m| {
            m.insert("schema_version", Value::Integer(SCHEMA_VERSION + 1));
        });

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Manifest(ref message)) if message.contains("schema version")
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_chunk_file_is_rejected() {
        let dir = scratch_dir("missing-chunk");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        fs::remove_file(dir.join("value-adam.chunk")).unwrap();

        assert!(matches!(load(&dir), Err(CheckpointError::Io { .. })));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_digest_for_the_wrong_chunk_is_rejected() {
        let dir = scratch_dir("swapped-digest");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        edit_manifest(&dir, |m| {
            let mut chunks = m.require_table("chunks").unwrap().clone();
            let value = chunks.get("value").unwrap().clone();
            chunks.insert("normalizer", value);
            m.insert("chunks", Value::Table(chunks));
        });

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Chunk { ref name, .. }) if name == "normalizer"
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn policy_shape_that_disagrees_with_its_weights_is_rejected() {
        let dir = scratch_dir("policy-shape");
        save(&dir, &agent(), &Rng::new(SeedStream::new(3, 0))).unwrap();
        edit_manifest(&dir, |m| {
            m.insert("policy_widths", widths_value(&[2, 3, 1]));
        });

        assert!(matches!(
            load(&dir),
            Err(CheckpointError::Invalid(ref message)) if message.contains("topology")
        ));
        let _ = fs::remove_dir_all(&dir);
    }
}
