## 1. OpenSpec and governance foundation

- [ ] 1.1 Finalize the proposal, design, and capability deltas for `finalize-archive-ready-repo`
- [ ] 1.2 Rewrite `CONTRIBUTING.md` around the current capability-based OpenSpec workflow
- [ ] 1.3 Refactor `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` to remove non-repository protocol noise and duplicated guidance

## 2. Public surface cleanup

- [ ] 2.1 Audit `docs/`, `changelog/`, and root markdown surfaces for duplicated or stale content
- [ ] 2.2 Rework `README.md` and `README.zh-CN.md` so they match the final archive-ready project positioning
- [ ] 2.3 Tighten `docs/.vitepress/config.mts`, docs home pages, and bilingual navigation around the same public story
- [ ] 2.4 Align GitHub About metadata, homepage, and topics with the finalized README/Pages wording

## 3. Engineering simplification and bug fixing

- [ ] 3.1 Audit `kernels/`, `utils/`, `tests/`, `benchmarks/`, and `examples/` for real bugs, validation drift, and edge-case inconsistencies
- [ ] 3.2 Implement root-cause fixes together with the affected tests and adjacent documentation
- [ ] 3.3 Normalize dependency and version policy across `pyproject.toml`, `requirements.txt`, `package.json`, `Makefile`, and contributor docs
- [ ] 3.4 Simplify `.github/workflows/*.yml` and adjacent repo templates to the minimum high-signal baseline

## 4. AI tooling, validation, and handoff

- [ ] 4.1 Review `.claude/**` and module-level `CLAUDE.md` files; delete, merge, or keep them based on repository-specific value
- [ ] 4.2 Add or refine machine-agnostic editor/LSP/tooling guidance only where it materially improves this repository
- [ ] 4.3 Run the validation baseline and fix any regressions that block archive-ready stability
- [ ] 4.4 Update change task state, summarize remaining risks, and leave a clean handoff trail for completion and archival
