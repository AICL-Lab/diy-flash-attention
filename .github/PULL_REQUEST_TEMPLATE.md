## Description
Briefly describe the changes in this PR.

## Related Issue
Closes # (issue number)

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] CI / Workflow improvement

## Spec Compliance
This project follows **Spec-Driven Development (SDD)**. Please confirm:
- [ ] Relevant spec documents in `/specs/` have been created or updated
- [ ] Code implementation matches the spec definitions
- [ ] Tests cover all acceptance criteria from the spec

## Testing
- [ ] CPU tests pass (`pytest tests/ -v -m "not cuda"`)
- [ ] Lint passes (`ruff check . && ruff format --check .`)
- [ ] Type checks pass (`mypy kernels utils`)
- [ ] New unit tests added (if applicable)

## Checklist
- [ ] My code follows the project's coding style
- [ ] I have added comments where necessary
- [ ] I have updated the documentation (if applicable)
- [ ] My changes generate no new warnings
