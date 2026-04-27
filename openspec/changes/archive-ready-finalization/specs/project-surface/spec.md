# project-surface Delta

## MODIFIED Requirements

### Requirement: Python Version Support
The project SHALL support Python 3.10 or higher, dropping support for Python 3.9.

**Before**: Python >= 3.9
**After**: Python >= 3.10
**Rationale**: Python 3.9 EOL (2025-10), CI not testing 3.9

#### Scenario: Python version requirement updated
- Given the project requires Python >= 3.9
- When the version requirement is updated to >= 3.10
- Then pyproject.toml reflects the new requirement
- And CI tests Python 3.10, 3.11, 3.12

### Requirement: pre-commit Hooks Version
The project MUST use current stable versions of pre-commit hooks.

**Before**: ruff v0.4.0, pre-commit-hooks v4.5.0
**After**: ruff v0.11.7, pre-commit-hooks v5.0.0
**Rationale**: Update to latest stable versions for improved linting

#### Scenario: pre-commit hooks updated
- Given the project uses outdated pre-commit hooks
- When hooks are updated to latest versions
- Then ruff is at v0.11.7
- And pre-commit-hooks is at v5.0.0

## REMOVED Requirements

### Requirement: requirements.txt
The project MUST NOT have a separate requirements.txt file.

**Rationale**: Consolidate dependency management to pyproject.toml only

#### Scenario: requirements.txt removed
- Given the project has a requirements.txt file
- When the file is removed
- Then all dependencies are managed through pyproject.toml

### Requirement: .githooks directory
The project MUST NOT have a .githooks directory.

**Rationale**: Consolidate to pre-commit framework only

#### Scenario: .githooks removed
- Given the project has a .githooks directory
- When the directory is removed
- Then git hooks are managed through pre-commit only

### Requirement: dev/ directory
The project MUST NOT have a dev/ directory.

**Rationale**: Contains only low-value temporary documentation

#### Scenario: dev/ removed
- Given the project has a dev/ directory
- When the directory is removed
- Then the repository has no temporary documentation clutter

### Requirement: changelog/archive/ directory
The project MUST NOT have a changelog/archive/ directory.

**Rationale**: Duplicate content already in CHANGELOG.md

#### Scenario: changelog/archive/ removed
- Given the project has a changelog/archive/ directory
- When the directory is removed
- Then all changelog history is in CHANGELOG.md
