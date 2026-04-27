## ADDED Requirements

### Requirement: Local workflow defaults are committed
The repository SHALL commit lightweight editor, LSP, and automation guidance that contributors can adopt without machine-specific secrets.

#### Scenario: Opening the repository in a supported editor
- **WHEN** a contributor opens the repository in an editor with workspace recommendations
- **THEN** the repository SHALL offer Python, Ruff, Markdown, YAML, and TOML guidance appropriate for this codebase

### Requirement: Automation remains lightweight
The repository SHALL document a lightweight local hook and validation path before broad cleanup changes are pushed.

#### Scenario: Preparing to push a cleanup change
- **WHEN** a contributor is ready to publish a non-trivial cleanup slice
- **THEN** the workflow SHALL identify the local checks and review pass expected before push or merge
