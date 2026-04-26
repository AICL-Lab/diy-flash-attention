## Context

The repository is already positioned as an educational, forward-only Triton FlashAttention project, and the existing capability-based OpenSpec specs validate cleanly. However, several cleanup seams remain:

1. The current stabilization change is complete, so a new broad cleanup run needs its own change trail instead of piggybacking on finished tasks.
2. Governance surfaces still disagree in important places. `CONTRIBUTING.md` points to a legacy `/specs` model, while `.github/copilot-instructions.md` contains non-repository protocol content that does not belong in this codebase.
3. The public documentation stack is directionally correct but still redundant. Docs-level changelog mirrors coexist with root changelog history, and some pages are more decorative than useful.
4. CI is not wildly overbuilt, but it still needs a final pass to ensure triggers, commands, and documented local workflow stay perfectly aligned.
5. The repository still needs a deliberate bug sweep and version-policy pass before it can be handed off as an archive-ready artifact.

The goal is not to expand scope. It is to finish the existing scope cleanly and make future maintenance boring.

## Goals / Non-Goals

**Goals:**

- Establish a fresh OpenSpec change for the final archive-readiness pass.
- Rebuild the control-document layer so OpenSpec, contributor guidance, and AI instructions all describe the same workflow.
- Reduce public docs to a high-signal bilingual surface with one clear story and one changelog authority.
- Align CI, dependency policy, GitHub metadata, and local workflow guidance with the repository's actual validation baseline.
- Find and fix real bugs or edge-case drift in the educational Python code paths without speculative expansion.
- Leave a clean handoff trail for the next agent/model to continue or finish the finalization work.

**Non-Goals:**

- Adding new Triton kernel features that expand the project's product scope.
- Turning the repository into a generic AI-tool showcase or multi-assistant configuration zoo.
- Introducing heavy release engineering, persistent MCP infrastructure, or large new CI matrices.
- Rebranding the project away from its current educational, forward-only FlashAttention focus.

## Decisions

### 1. Create one new finalization change instead of reopening completed work

- **Decision**: Use a new change, `finalize-archive-ready-repo`, as the sole umbrella for this endgame cleanup.
- **Why**: The previous stabilization thread is already complete. Reusing it would blur historical accountability and make archive handoff harder to reason about.
- **Alternative considered**: Keep editing directly or reopen `stabilize-project-for-archive`.
- **Why not**: That would weaken OpenSpec discipline and make it unclear which tasks belong to which cleanup wave.

### 2. Treat governance drift as a first-class defect

- **Decision**: Rewrite or remove mismatched control documents before large implementation edits.
- **Why**: Misleading governance docs cause repeated low-quality changes. They are not ancillary; they actively shape how later work gets done.
- **Alternative considered**: Fix code/docs first and clean governance later.
- **Why not**: That would keep bad instructions in place during the most invasive part of the cleanup.

### 3. Prefer deletion and consolidation over additive documentation

- **Decision**: Collapse duplicated changelog/document surfaces and keep only pages that explain or route users effectively.
- **Why**: The repository is in a finishing phase. More documents are valuable only when they reduce ambiguity or preserve unique information.
- **Alternative considered**: Preserve redundant pages but polish them.
- **Why not**: That increases maintenance burden and keeps multiple stale truth sources alive.

### 4. Keep GitHub Pages as a focused showcase, not a second documentation universe

- **Decision**: Refine the existing VitePress site around strong entrypoints, bilingual routing, and clear cross-links back to code and specs.
- **Why**: The current site already has a usable showcase direction; the right move is to tighten information architecture, not build a bigger marketing layer.
- **Alternative considered**: Perform a full visual/structural rewrite.
- **Why not**: That would spend effort on presentation churn instead of clarity and alignment.

### 5. Keep engineering automation minimal and exactly mirrored to local commands

- **Decision**: The validation baseline remains OpenSpec validation, docs build, lint, type-check, and CPU-safe tests. CI should do those well and little else.
- **Why**: The repository is archive-bound. High-signal checks matter; elaborate automation mostly creates future maintenance noise.
- **Alternative considered**: Add more matrices, stronger release automation, or broader formatting/lint layers.
- **Why not**: That would exceed the project's maintenance needs.

### 6. Make AI tooling guidance explicit, repository-specific, and narrow

- **Decision**: Retain only the AI guidance that helps with this repository's Python + VitePress + OpenSpec stack, and explicitly document trade-offs for Copilot, Claude, Codex/Opencode, `gh`, optional MCP, and CLI skills.
- **Why**: Generic instruction layers have already introduced drift and non-functional process noise.
- **Alternative considered**: Keep many layered assistant configs for maximum optionality.
- **Why not**: Optionality without curation becomes boilerplate and contradicts the archive-ready goal.

## Risks / Trade-offs

- **[Risk] Deleting docs removes useful historical context** → Preserve unique history in one changelog authority and redirect readers from removed/merged pages through adjacent docs.
- **[Risk] Governance cleanup conflicts with user-local expectations** → Keep committed instructions repo-specific and machine-agnostic; avoid committing personal or secret-bearing config.
- **[Risk] Bug sweep expands into feature work** → Limit fixes to defects, validation drift, and behavior already implied by current scope/specs.
- **[Risk] GitHub metadata changes drift again** → Change About/homepage/topics only after README and Pages wording are finalized.
- **[Risk] Broad cleanup loses handoff clarity** → Keep the change tasks explicit and update task status as each phase lands.

## Migration Plan

1. Create the new finalization change and update capability deltas that describe the new governance/public-surface/workflow expectations.
2. Rewrite governance surfaces and contributor instructions so the repo has one trustworthy workflow description.
3. Prune and realign public docs, changelog handling, README copy, and GitHub Pages information architecture.
4. Sweep implementation/test/config surfaces for real bugs and version-policy drift; fix them together with adjacent docs/specs.
5. Tighten GitHub workflows and remote metadata only after local content and commands are stable.
6. Run the full validation matrix, update tasks/handoff notes, and leave the change in a clean state for completion and archival.

Rollback is straightforward because the work is mostly file- and config-based: revert the change set and restore the previous docs/config layering if needed.

## Open Questions

- Whether any module-level `CLAUDE.md` files contain enough unique guidance to justify keeping them after the consolidation pass.
- Whether docs-level changelog pages should be removed entirely or retained as thin routing pages to the root changelog authority.
- Whether `gh` authentication is already available for live About/topic updates, or whether those commands should be staged locally first.
