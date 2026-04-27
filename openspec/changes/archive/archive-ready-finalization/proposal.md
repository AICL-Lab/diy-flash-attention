# Proposal: Final Archive-Ready Repository Cleanup

## Summary

Complete the final cleanup and stabilization of diy-flash-attention repository to achieve archive-ready state.

## Motivation

The repository has accumulated several technical debts and inconsistencies that need to be resolved before it can be considered fully stable and ready for archival handoff:

1. **Redundant directories**: `dev/` and `changelog/archive/` contain low-value or duplicate content
2. **OpenSpec state inconsistency**: One completed change not archived, one empty change not cleaned
3. **Outdated tooling**: pre-commit hooks versions are ~1 year behind
4. **Version conflicts**: Python version declaration inconsistent with CI testing
5. **Duplicate configurations**: requirements.txt duplicates pyproject.toml, .githooks duplicates pre-commit

## Scope

- Directory cleanup (`dev/`, `changelog/archive/`, `.githooks/`, `requirements.txt`)
- OpenSpec change archival and cleanup
- Tooling updates (pre-commit hooks, Python version)
- Documentation alignment (CONTRIBUTING.md)
- Version bump to 1.0.4

## Success Criteria

- All redundant directories removed
- OpenSpec shows no active changes (or only this one)
- All lint/typecheck/tests pass
- CI pipeline runs successfully
- Documentation builds correctly

## Timeline

Single execution session.

## Risks

- Low: Python version change may affect users on Python 3.9
- Mitigation: v1.0.3 remains available for Python 3.9 users

## Dependencies

None.
