## Appendix D — Rust extensibility patterns

This appendix describes implementation choices that may realize Sections 7 and 9. It is advisory: Section 9.6 and the decision records in Appendix C govern, and no pattern below authorizes dynamic code loading or changes the timing rules in Section 5.

### D.1 Dispatch and composition

Rust generics are monomorphized for concrete types and provide compile-time method dispatch. This is the preferred default for a control-path implementation and shared interfaces whose concrete type is known when the executable is built; it trades runtime indirection for compile-time and binary-size growth. Benchmark both effects rather than assuming the resulting binary is smaller or faster.

A closed enum is appropriate when the implementations are known and finite at build time. `enum_dispatch` is a candidate macro to forward a trait-like API across enum variants; compare it with a manually authored enum and keep the set auditable. It is not an open plugin registry.

Trait objects can simplify heterogeneous experiment composition, but their use is limited to offline training, evaluation and tooling. Name-keyed registries may resolve a manifest to an implementation once at process startup from code already linked into that executable. They MUST NOT register or unregister methods during a run and MUST NOT be queried on a control tick. `inventory` distributes registrations from linked crates through constructor-based initialization; `linkme` gathers const static entries into linker sections and therefore depends on supported linker/platform behavior. Qualify initialization, dead-stripping, target support and deterministic duplicate-name rejection before selecting either mechanism.

`typetag` combines trait-object serialization with distributed registration, relying on `inventory` and constructor initialization; it constructs a tag-to-deserializer map on first trait-object deserialization. `erased-serde` supplies object-safe Serde interfaces. These are possible offline interchange mechanisms, not canonical schema or migration systems. Prefer a versioned explicit enum for a stable set of serialized variants, and never deserialize tagged trait objects on a control path or treat one as executable plugin loading.

### D.2 Robot and experience data layout

The robot descriptor and transition schema are authored Rust types independent of the chosen serializer. Stable typed IDs, explicit schema versions, namespaced extensions and migration rules preserve meaning as embodiments and methods grow. Registries describe capabilities; they do not replace validation of typed fields or grant authority to unknown metadata.

SoA chunks place each field in a contiguous column and can make batched sampling and vectorized field access efficient; AoS or tiled AoSoA may be better for other access patterns. Benchmark representative observation/action dimensions, batch sizes, cache behavior, gather cost and memory use before freezing a layout. Keep serialization chunks bounded and versioned as required by Sections 7.5 and 7.6.

A fixed-capacity ring gives bounded memory and predictable overwrite behavior, but the buffer contract must additionally define capacity exhaustion, generation IDs, wraparound and access by sampled transition index. A simple FIFO/SPSC ring is not a complete replay store: prioritized or sequence sampling needs separate index structures, and sampler state/RNG must be checkpointed if exact continuation is required. Keep sampler traits separate from storage so uniform, prioritized, sequence and goal-relabeling methods can evolve without changing the transition format.

### D.3 Qualification references

The mechanism descriptions in Appendix B.2.14 link to the Rust Book and crate documentation for `enum_dispatch`, `inventory`, `linkme`, `typetag`, `erased-serde` and `ringbuffer`. Those references are research starting points, not endorsements. Before adoption, record the exact version, license, platform and linker behavior, feature/dependency closure, startup behavior, benchmark results, schema compatibility and removal cost under Appendix B.6.

## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 2 `[x]`, 2 `[~]`, 2 `[ ]`, 0 `[?]` across 6 items. This appendix is advisory; the checklist records the constraints it places on code.

- [x] D.1-a No registry, plugin or dynamic code loading — Evidence: a case-insensitive `grep` of `*.rs`, `*.toml` and `*.nix` outside `target/` for `enum_dispatch`, `linkme`, `typetag`, `erased-serde`, `libloading` and `dlopen` finds nothing, and `inventory` appears in no manifest; `[workspace.dependencies]` in `Cargo.toml` is empty.
- [x] D.1-b Registries do not mutate during a run or get queried on a control tick — Evidence: no registry exists (see 9.6.4 and C-20).
- [ ] D.1-c Benchmark generics against trait objects before claiming size or speed — Evidence: `docs/benchmarks/` is empty and no crate has a `benches/` directory; no size or speed claim about dispatch is made anywhere, so nothing is in breach. Note: becomes due when a hot-path trait object is introduced.
- [~] D.2-a Ring buffer defines capacity exhaustion, generation IDs and wraparound — Evidence: `crates/robot-learn/src/rollout.rs` `RolloutBuffer` (line 48) is fixed-capacity, reserves up front, refuses a push past capacity with `CapacityExhausted` (line 86) rather than wrapping, and every transition carries a `Generation`; `crates/robot-types/src/transition.rs` line 583 declares the `BoundedRing` policy but no implementation consumes it (`grep BoundedRing` hits only that file). Note: exhaustion and generation are defined; an overwriting ring with wraparound semantics does not exist.
- [~] D.2-b Sampler traits separate from storage — Evidence: no sampler trait exists (see 9.3.2); the stores and the sampler are not separated because the prioritized sampler is absent.
- [ ] D.3-a Each adopted mechanism has a B.6 qualification record — Evidence: no mechanism from this list is adopted, so the record is not yet needed; the gap is recorded as C-20 and C-21 for future adoption.
