# Changelog: 2025-02-27 Python Version Fix

**Date**: 2025-02-27  
**Type**: Bug Fix  
**Version**: 1.0.2

## Overview

Fixed Python version requirement to match actual code usage of Python 3.9+ syntax features.

## Problem

The codebase uses Python 3.9+ type annotation syntax:
- `tuple[int, int]` instead of `Tuple[int, int]`
- `list[dict]` instead of `List[Dict]`
- `dict[str, Any]` instead of `Dict[str, Any]`

However, `pyproject.toml` specified `requires-python = ">=3.8"`, causing potential runtime errors.

## Changes

### pyproject.toml

```diff
- requires-python = ">=3.8"
+ requires-python = ">=3.9"
```

- Added Python 3.12 classifier
- Updated mypy target version to 3.9

### CI Workflow

```diff
- python-version: ['3.8', '3.9', '3.10', '3.11']
+ python-version: ['3.9', '3.10', '3.11', '3.12']
```

## Impact

- ✅ Prevents installation on incompatible Python versions
- ✅ Accurate version metadata
- ✅ CI tests correct Python versions

## References

- PEP 585: Type Hinting Generics In Standard Collections
- Python 3.9 Release Notes
