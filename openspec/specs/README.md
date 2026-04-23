# OpenSpec Specifications

`openspec/specs/` stores the repository's **capability specifications**. Each capability lives in its own folder and exposes a single normative file:

```text
openspec/specs/<capability>/spec.md
```

This repository uses the OpenSpec `spec-driven` schema:

- `openspec/specs/<capability>/spec.md` stores the **current archived requirements**
- `openspec/changes/<change>/specs/<capability>/spec.md` stores the **delta for an active change**

## Current capabilities

| Capability | Purpose |
| --- | --- |
| `flashattention-kernels` | Normative behavior for Triton matmul, FlashAttention forward pass, architecture adaptation, benchmarks, and correctness checks |
| `project-governance` | OpenSpec workflow, review discipline, AI-tool coordination, and archive-ready scope management |
| `project-surface` | README, GitHub Pages, and GitHub repository metadata requirements |
| `engineering-workflow` | CI, hooks, local editor/LSP baseline, and AI tooling guidance |

## How this differs from the old layout

The repository previously kept PRD/RFC/testing material under directories such as `product/`, `rfc/`, and `testing/`. Those documents were useful source material, but they did **not** match the active OpenSpec schema and were not treated as first-class capabilities.

This directory now follows the stricter capability model:

- use **proposal.md** for *why*
- use **design.md** for *how*
- use **spec.md** for *what the system SHALL do*
- use **tasks.md** for executable implementation slices

## Workflow

1. Start with `openspec list --json` and select or create a change.
2. Write `proposal.md`, then `design.md`, then change-delta specs under `openspec/changes/<change>/specs/...`.
3. Apply tasks from the active change.
4. Archive the change once tasks are done and fold the final spec deltas back into `openspec/specs/`.

## Project note

This repository is intentionally optimized for **archive-ready stability**. Specs should stay concise, testable, and tightly coupled to the actual Triton kernels, docs, GitHub Pages surface, and engineering workflow that ship with this project.
