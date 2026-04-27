## 1. OpenSpec normalization

- [x] 1.1 Replace the legacy `openspec/specs/{product,rfc,testing,api}` layout with capability-based `spec.md` files.
- [x] 1.2 Write the proposal, design, and implementation plan for `stabilize-project-for-archive`.

## 2. Governance and AI workflow

- [x] 2.1 Create root `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` as the single workflow authority.
- [x] 2.2 Add project-level editor, LSP, and local hook guidance that fits the repository's Python + VitePress stack.

## 3. Public project surfaces

- [x] 3.1 Rewrite `README.md` and `README.zh-CN.md` so they reflect the final OpenSpec layout and current project positioning.
- [x] 3.2 Redesign VitePress home, config, and public metadata so GitHub Pages works as a project showcase.
- [x] 3.3 Update GitHub About metadata, homepage, and topics with `gh`.

## 4. Engineering simplification and validation

- [x] 4.1 Simplify CI/workflows, hook configuration, Makefile commands, and dependency policy around high-signal checks only.
- [x] 4.2 Run OpenSpec validation, lint, type-checking, docs build, and CPU-safe tests; fix regressions that block archive-ready stability.
