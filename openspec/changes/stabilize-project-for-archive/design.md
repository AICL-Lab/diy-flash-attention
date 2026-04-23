## Context

The repository currently mixes three different models of project governance:

1. A partially migrated OpenSpec tree that does not follow the active `spec-driven` schema.
2. Legacy-style narrative docs (`product`, `rfc`, `testing`) that are useful as source material but are not valid OpenSpec capabilities.
3. Several public/project surfaces that describe the project differently: README, VitePress, GitHub Pages, CI, and GitHub About metadata.

The project goal is no longer open-ended expansion. It is to stabilize the repository into a high-quality educational artifact that can be maintained lightly and archived gracefully.

## Goals / Non-Goals

**Goals:**

- Restore strict OpenSpec compatibility with capability-based specs.
- Encode one lightweight but enforceable workflow for OpenSpec, AI agents, review, and branch hygiene.
- Align README, docs, GitHub Pages, and GitHub About metadata around the same project positioning.
- Keep only automation that materially protects quality for this repository.
- Provide project-level local development templates that improve consistency without forcing machine-specific secrets or heavyweight daemons into git.

**Non-Goals:**

- Adding new kernel features unrelated to repository stabilization.
- Rewriting the core Triton kernels unless validation exposes a real bug.
- Introducing heavy infrastructure such as persistent MCP stacks, multi-stage release engineering, or broad contributor process bureaucracy.
- Turning the project into a general GPU programming portal beyond the FlashAttention learning focus.

## Decisions

### 1. Use capability-based OpenSpec specs

The repository will migrate from `product/rfc/testing/api` pseudo-capabilities to real capability folders under `openspec/specs/<capability>/spec.md`.

- **Why**: This matches the active OpenSpec schema and makes validation, proposing, applying, and archiving reliable.
- **Alternative considered**: Keep the existing hierarchy and treat it as documentation only.
- **Why not**: OpenSpec currently sees those folders as empty capabilities, which is structurally misleading and brittle.

### 2. Keep one long-lived stabilization change

This cleanup will be driven by a single change, `stabilize-project-for-archive`, instead of many small changes.

- **Why**: The work is intentionally cross-cutting and endgame-oriented. Splitting it into many speculative changes adds overhead and makes later autopilot runs harder to reason about.
- **Alternative considered**: Separate changes for docs, workflows, tooling, and metadata.
- **Why not**: That would duplicate context and re-open the same architectural questions across several change threads.

### 3. Separate committed policy from machine-local activation

The repository will commit shared instructions, recommendations, and lightweight templates, but not machine-specific credentials or personalized global configs.

- **Why**: The project needs consistent defaults without making assumptions about a contributor's editor, authentication, or host environment.
- **Alternative considered**: Commit richer per-tool config for every supported AI assistant and local service.
- **Why not**: That increases maintenance cost and quickly drifts into generic boilerplate.

### 4. Prefer high-signal automation over maximal automation

The repository will keep only the checks that meaningfully protect current project quality: OpenSpec validation, docs build, lint, type-checking, and CPU-safe tests.

- **Why**: The repository is heading toward archive-ready stability, not ongoing platform expansion.
- **Alternative considered**: Add more workflows, more matrixes, and more specialized guards.
- **Why not**: More automation would mostly increase maintenance noise without protecting proportionate value.

### 5. Treat GitHub Pages as a project showcase, not a README mirror

The VitePress site will emphasize why this project exists, what readers can learn, and where to start.

- **Why**: The repository needs a strong public landing page that complements GitHub instead of duplicating it.
- **Alternative considered**: Keep Pages as a polished README clone.
- **Why not**: That adds little discovery value and weakens the repository's external presentation.

## Risks / Trade-offs

- **[Risk] Existing uncommitted work overlaps with this cleanup** → Review current diffs before editing and integrate useful pieces instead of bluntly reverting them.
- **[Risk] Spec migration may break references from older docs** → Update README/docs/governance references in the same cleanup wave.
- **[Risk] Tooling guidance becomes generic** → Keep every workflow rule tied to this repository's Triton, CUDA, docs, and OpenSpec realities.
- **[Risk] GitHub metadata changes drift from repository contents** → Update About/homepage/topics only after docs and README positioning are finalized.
- **[Risk] Local editor/LSP setup becomes too opinionated** → Commit recommendations and workspace settings only; leave personal/global choices to contributors.

## Migration Plan

1. Create valid capability specs and the stabilization change artifacts.
2. Replace legacy governance docs with root-level, project-specific guidance.
3. Realign README, docs, and Pages positioning.
4. Simplify CI/hooks/local workflow files.
5. Update GitHub About/homepage/topics with `gh`.
6. Run validation (OpenSpec, lint, type-check, docs build, CPU-safe tests) and fix regressions.

Rollback is straightforward because the work is file- and config-based: revert the changed docs/configs and restore the previous OpenSpec layout if needed.

## Open Questions

- Whether a tiny repository-local hook script should complement `pre-commit`, or whether `pre-commit install` alone is sufficient.
- Whether any MCP integration is justified beyond documentation of trade-offs.
