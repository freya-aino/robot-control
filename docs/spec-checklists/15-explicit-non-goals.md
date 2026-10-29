## 15. Explicit non-goals for the first release

The first release does not promise universal zero-shot robot control, production autonomous boxing, full video-to-motion reconstruction, an unverified Rust GPU simulator, arbitrary online policy learning on hardware, a language model in the servo loop, an agent-orchestrated strategy, personality or operations layer, a robot-hosted agent runtime, an agent-selected action or an agent-owned safety or promotion gate, durable agent-run training campaigns, or reproduction of every named paper. It delivers the contracts and measured foundations needed to add those capabilities deliberately. Development-time automation, offline sampling assistance and the dependency knowledge base of Section 2.3.1 are tooling rather than release capabilities: no release claim rests on them, and none of them is a promise of this release either.


## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 0 `[~]`, 0 `[ ]`, 1 `[?]` across 5 items.

- [x] 15.1 No universal zero-shot control claim — Evidence: `docs/ledger/evidence-ledger.md` closing note: no row licenses a claim beyond the listed host and embodiment.
- [x] 15.2 No language model in the servo loop — Evidence: a case-insensitive grep of `crates/` for `llm`, `language model`, `openai`, `anthropic` and `agent runtime` returned no match in any dependency or source file.
- [x] 15.3 No robot-hosted agent runtime or agent-selected action — Evidence: the same grep found no agent crate; `crates/robot-persona/` is outside `robot-control` (see 6.4).
- [x] 15.4 No durable agent-run training campaigns — Evidence: no agent-run configuration or campaign directory was found in `crates/` or `docs/`.
- [?] 15.5 No reproduction of every named paper — a documentation scope rule, not audited against the ledger in this pass.
