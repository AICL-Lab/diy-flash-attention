## 1. Directory Cleanup

- [x] 1.1 Delete `dev/` directory (5 low-value temp docs)
- [x] 1.2 Delete `changelog/archive/` directory (7 duplicate archived files)
- [x] 1.3 Delete `requirements.txt` (consolidate to pyproject.toml)
- [x] 1.4 Delete `.githooks/` directory (consolidate to pre-commit)

## 2. OpenSpec Change Management

- [x] 2.1 Archive `stabilize-project-for-archive` (completed but not archived)
- [x] 2.2 Delete `finalize-archive-ready-repo` (empty, overlaps with this change)
- [x] 2.3 Create `archive-ready-finalization` change

## 3. Configuration Updates

- [x] 3.1 Update ruff version: v0.4.0 → v0.11.7
- [x] 3.2 Update pre-commit-hooks version: v4.5.0 → v5.0.0
- [x] 3.3 Update Python version: >=3.9 → >=3.10
- [x] 3.4 Update ruff target-version: py39 → py310
- [x] 3.5 Update mypy python_version: 3.9 → 3.10
- [x] 3.6 Update Makefile hooks-install (remove .githooks)
- [x] 3.7 Update Makefile install (remove requirements.txt)

## 4. Documentation Updates

- [x] 4.1 Rewrite CONTRIBUTING.md (remove obsolete directory refs)
- [x] 4.2 Update CHANGELOG.md (add v1.0.4 entry)
- [x] 4.3 Update version in pyproject.toml: 1.0.3 → 1.0.4

## 5. Validation

- [ ] 5.1 Run `make lint`
- [ ] 5.2 Run `make typecheck`
- [ ] 5.3 Run `make test-cpu`
- [ ] 5.4 Run `make docs`
- [ ] 5.5 Run `make validate-openspec`
- [ ] 5.6 Run `make hooks-run`

## 6. GitHub Sync

- [ ] 6.1 Push all changes
- [ ] 6.2 Clean local branches
- [ ] 6.3 Create v1.0.4 tag
- [ ] 6.4 Push tag

## 7. Finalization

- [ ] 7.1 Archive this change
- [ ] 7.2 Verify CI passes
