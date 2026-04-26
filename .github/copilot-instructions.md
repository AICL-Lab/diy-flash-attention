# Copilot Instructions

This repository is an **educational, forward-only Triton FlashAttention project** being finalized for archive-ready maintenance. Keep the repository small, coherent, and easy to trust.

## OpenSpec-first workflow

1. Run `openspec list --json` before non-trivial work.
2. Read the active change artifacts plus the affected `openspec/specs/<capability>/spec.md` files.
3. If no relevant active change exists, create/propose one before editing.
4. Implement against `tasks.md`, update adjacent docs/tests/specs together, and archive completed changes instead of leaving stale active work behind.

## Repository priorities

- Preserve the **hands-on, forward-only** FlashAttention learning scope.
- Keep README, bilingual docs, GitHub Pages, and GitHub About metadata aligned.
- Prefer deleting stale docs, duplicated changelog surfaces, and workflow clutter over adding scaffolding.
- Keep CI, hooks, and tool guidance lightweight and high-signal.

## Editing rules

- Avoid generic contributor or AI boilerplate.
- Do not hard-code a completed change name into long-lived guidance.
- Update nearby specs, docs, tests, and metadata when behavior or messaging changes.
- Use review for broad or risky cleanup slices before merge or archival handoff.

## Validation baseline

- `make lint`
- `make typecheck`
- `pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py`
- `npm run docs:build`
- `openspec validate`

## Tooling stance

- Use Copilot for focused edits and GitHub-native flows.
- Use Claude for cross-file reasoning, docs/governance cleanup, and OpenSpec refinement.
- Use Codex/opencode-style tools for crisp execution once tasks are clear.
- Use `gh` for repository metadata updates after README/Pages wording is finalized.
- Treat MCP as optional; only introduce it when it clearly reduces recurring maintenance pain in this repo.
