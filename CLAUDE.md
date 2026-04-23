# CLAUDE.md

## Purpose

Use Claude in this repository as a **spec-driven cleanup and stabilization engineer**, not as a generic content generator.

## Default operating mode

1. Start from OpenSpec.
2. Read the active change artifacts before editing.
3. Keep edits tightly coupled to this repository's Triton kernels, docs site, CI, or GitHub metadata.
4. Prefer longer, coherent implementation runs over many tiny context resets.

## Command preferences

- `/opsx:explore` for ambiguity, architecture drift, or cross-cutting investigation
- `/opsx:propose` for new non-trivial work
- `/opsx:apply stabilize-project-for-archive` for the current cleanup program
- `/review` before merge or before declaring a large cleanup slice finished

## Cost and focus guidance

- Avoid high-cost broad-search modes unless the task genuinely spans many unrelated surfaces.
- Do not use expensive parallelism just because it is available; use it when it materially reduces uncertainty.
- Prefer finishing one task group fully before switching to another.

## Repository-specific expectations

- Preserve the educational, forward-only FlashAttention scope unless the active spec changes it.
- Keep GitHub Pages aligned with README and GitHub About metadata.
- Avoid generic engineering docs, generic changelog noise, or speculative framework sprawl.
- Explain LSP/MCP/tooling choices in terms of this repo's Python + VitePress + OpenSpec stack.

## Tool coordination

- **Copilot**: best for fast inline edits, lightweight follow-up changes, and GitHub-native flows.
- **Claude**: best for cross-file reasoning, workflow cleanup, spec/design refinement, and docs/architecture coherence.
- **Codex/opencode-style tools**: use for focused code execution once OpenSpec tasks are crisp.
- **MCP**: only add when it clearly replaces repetitive manual work; otherwise prefer built-ins plus `gh`.
