## Description
Briefly describe the changes in this PR.

## Related OpenSpec Change
- Change: `openspec/changes/<change-name>/`
- Issue / discussion (optional): Closes #

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] CI / Workflow improvement

## OpenSpec Compliance
- [ ] Relevant change artifacts and/or affected `openspec/specs/<capability>/spec.md` files were reviewed or updated
- [ ] Code/docs/config changes match the OpenSpec intent
- [ ] Adjacent tests and documentation were updated when behavior or messaging changed

## Testing
- [ ] `make lint`
- [ ] `make typecheck`
- [ ] `pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py`
- [ ] `npm run docs:build` (if docs / Pages / metadata-adjacent files changed)
- [ ] New unit tests added (if applicable)

## Checklist
- [ ] My code follows the project's coding style
- [ ] I have updated the documentation (if applicable)
- [ ] I used review for a broad or risky cleanup slice
