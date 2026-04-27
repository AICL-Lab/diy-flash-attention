## Why

The repository has drifted away from a clean OpenSpec-driven workflow. Its specs, AI guidance, docs, GitHub Pages, and automation are partially modernized but still inconsistent, which makes endgame maintenance noisy and unreliable.

This change consolidates the project into an archive-ready baseline: one clear OpenSpec workflow, one coherent public narrative, and one lightweight engineering setup that fits an educational Triton/FlashAttention repository.

## What Changes

- Replace the legacy `openspec/specs/{product,rfc,testing,api}` layout with capability-based specs that OpenSpec can actually validate and evolve.
- Create a finalization-oriented OpenSpec change package that future GLM/autopilot sessions can follow without guessing scope.
- Add a single source of truth for project governance across `AGENTS.md`, `CLAUDE.md`, Copilot instructions, and local editor/tooling guidance.
- Reposition README, bilingual docs, GitHub Pages, and GitHub repository metadata around the project's real value: learning Triton by implementing FlashAttention from scratch.
- Simplify workflows, hooks, and local quality gates so the repository keeps only high-signal automation.

## Capabilities

### New Capabilities

- `flashattention-kernels`: Normative requirements for the educational Triton kernels, architecture adaptation helpers, benchmarks, and correctness surface.
- `project-governance`: Project-specific OpenSpec workflow, AI tool coordination, branch/review discipline, and archive-ready scope control.
- `project-surface`: Requirements for README, GitHub Pages, and GitHub repository metadata to tell one coherent story.
- `engineering-workflow`: Requirements for CI, hooks, local editor/LSP setup, and AI-tooling integration with minimal maintenance overhead.

### Modified Capabilities

- None.

## Impact

- `openspec/` structure, change artifacts, and spec organization
- Root governance docs and AI instruction files
- README files, docs content, VitePress config, and GitHub Pages assets
- GitHub Actions, hooks, Makefile, and local development templates
- GitHub repository About/homepage/topics managed through `gh`
