# Design: Final Archive-Ready Repository Cleanup

## Architecture Impact

No changes to the core architecture. This is a maintenance and cleanup change only.

## Affected Components

| Component | Change Type | Impact |
|-----------|-------------|--------|
| `dev/` | Delete | Remove 5 low-value temp docs |
| `changelog/archive/` | Delete | Remove 7 duplicate archived files |
| `requirements.txt` | Delete | Consolidate to pyproject.toml |
| `.githooks/` | Delete | Consolidate to pre-commit |
| `.pre-commit-config.yaml` | Update | ruff v0.11.7, hooks v5.0.0 |
| `pyproject.toml` | Update | Python >=3.10, version 1.0.4 |
| `Makefile` | Update | Remove .githooks ref, update install |
| `CONTRIBUTING.md` | Rewrite | Remove obsolete directory refs |
| `CHANGELOG.md` | Update | Add v1.0.4 entry |

## Implementation Approach

### Phase 1: Directory Cleanup
1. Delete `dev/` directory
2. Delete `changelog/archive/` directory
3. Delete `requirements.txt`
4. Delete `.githooks/` directory

### Phase 2: OpenSpec Cleanup
1. Archive `stabilize-project-for-archive`
2. Delete `finalize-archive-ready-repo`
3. Create `archive-ready-finalization` change

### Phase 3: Configuration Updates
1. Update pre-commit hook versions
2. Update Python version declarations
3. Update Makefile hooks-install target

### Phase 4: Documentation
1. Rewrite CONTRIBUTING.md
2. Update CHANGELOG.md for v1.0.4

### Phase 5: Validation
1. Run lint, typecheck, test-cpu
2. Build docs
3. Validate OpenSpec
4. Verify CI passes

## Rollback Plan

All changes are reversible via git. Each phase can be committed separately for easy rollback if needed.

## Testing Strategy

- Run existing test suite (CPU-safe path)
- Verify docs build
- Run pre-commit on all files
- Check CI pipeline after push
