# AGENTS.md

This repository is an **educational Triton/FlashAttention project** that is being stabilized for archive-ready maintenance. Every change should make the repository clearer, leaner, and easier to trust.

## Workflow authority

Use OpenSpec as the only change-management system for non-trivial work:

1. `openspec list --json` to inspect active changes and current capabilities.
2. Use `explore` or `/opsx:explore` to investigate ambiguous work.
3. Use `propose` or `/opsx:propose` before starting new non-trivial work.
4. Use `apply` or `/opsx:apply <change>` to execute tasks from the selected change.
5. Use `archive` or `/opsx:archive <change>` once tasks are complete.

For repository-wide cleanup, continue the active change `stabilize-project-for-archive` unless there is a strong reason to split scope.

## What matters in this repo

- **Core code**: `kernels/`, `utils/`, `tests/`, `benchmarks/`, `examples/`
- **Specs**: `openspec/specs/<capability>/spec.md`
- **Change work**: `openspec/changes/<change>/`
- **Public surfaces**: `README.md`, `README.zh-CN.md`, `docs/`, GitHub About metadata

When one of these surfaces changes, update the adjacent ones that explain the same behavior. Do not let README, specs, docs, or CI drift apart.

## Project-specific rules

- Keep the project scoped as a **forward-only educational implementation** unless the active change explicitly expands scope.
- Prefer deleting stale or generic docs over polishing low-value duplication.
- Keep workflow docs and AI instructions **specific to this repository**. Avoid copied boilerplate.
- Prefer one active cleanup thread over many long-lived branches or unmerged cloud/local variations.
- Before merge or archival handoff for any non-trivial change, run a review pass such as `/review`.

## Validation baseline

Use the existing project commands and keep checks lightweight:

- `make lint`
- `make typecheck`
- `pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py`
- `npm run docs:build`
- `openspec validate`

Fix real failures. Do not add new automation tools unless the repository truly needs them.

## Tooling stance

- Prefer built-in repo tools, OpenSpec CLI, and `gh` over extra services.
- Use subagents/review models for wide or high-risk changes, not for every small edit.
- Treat MCP as optional. Do not add repo-local MCP config unless it provides recurring value that beats built-in tooling.
