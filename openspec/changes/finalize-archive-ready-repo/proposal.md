## Why

The repository already completed one stabilization pass, but it still carries visible drift across governance docs, duplicated public documentation, AI instruction layers, dependency/version policy, and GitHub automation. A final cleanup change is needed now so the project can reach an archive-ready baseline with one coherent workflow, one coherent public narrative, and a low-noise engineering surface for the next handoff.

## What Changes

- Create a final archive-readiness cleanup thread that replaces ad hoc follow-up edits with one explicit OpenSpec change.
- Rewrite repository governance and contributor guidance so every control document reflects the current capability-based OpenSpec workflow and repository-specific AI/tooling expectations.
- Remove or consolidate duplicated, stale, or low-value documentation across `docs/`, `changelog/`, and root guidance surfaces.
- Tighten README, bilingual docs, GitHub Pages, and GitHub About metadata so they present the same educational forward-only Triton/FlashAttention story.
- Simplify dependency/version policy, CI triggers, and GitHub workflow logic around the existing high-signal validation baseline.
- Audit the Python surfaces for real bugs and edge-case drift, then fix defects together with tests and adjacent docs.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `project-surface`: tighten the public docs contract so README, bilingual docs, changelog handling, GitHub Pages, and GitHub About stay synchronized and avoid duplication.
- `engineering-workflow`: strengthen requirements for version policy coherence, high-signal CI triggers, editor/LSP guidance, and repository-specific AI/tooling trade-off documentation.
- `project-governance`: strengthen the OpenSpec-first governance contract so repository instructions stay specific, broad cleanup work keeps an explicit active change, and handoff state remains clear.

## Impact

- `openspec/changes/` and selected capability specs under `openspec/specs/`
- Root governance documents: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `CONTRIBUTING.md`
- Public surfaces: `README.md`, `README.zh-CN.md`, `docs/**`, `changelog/**`, GitHub Pages assets
- Engineering/configuration files: `.github/workflows/*.yml`, `pyproject.toml`, `requirements.txt`, `package.json`, `Makefile`
- Python implementation and tests under `kernels/`, `utils/`, `tests/`, `benchmarks/`, and `examples/`
