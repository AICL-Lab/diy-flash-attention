## Purpose

Define the lightweight engineering baseline for CI, hooks, local editor/LSP guidance, and AI-tool coordination in this Python + VitePress repository.

## Requirements

### Requirement: Quality gates stay high-signal
The repository SHALL keep only automation that materially protects the current Python + Triton + VitePress codebase.

#### Scenario: Running continuous integration
- **WHEN** CI executes for repository changes
- **THEN** it SHALL validate OpenSpec structure, docs buildability, code style, type safety, and CPU-safe tests without adding redundant workflows

### Requirement: Hook installation is lightweight
The repository SHALL provide a lightweight local hook path for formatting and basic hygiene checks before changes are pushed.

#### Scenario: Enabling local hooks
- **WHEN** a contributor opts into repository hooks
- **THEN** the installation path SHALL rely on lightweight project-managed commands rather than hidden machine-specific scripts

### Requirement: Editor and LSP guidance is shared
The repository SHALL commit workspace guidance for Python, Markdown, TOML, YAML, and JSON editing so contributors get consistent diagnostics.

#### Scenario: Configuring a supported editor
- **WHEN** a contributor opens the repository in an editor that supports workspace recommendations
- **THEN** the repository SHALL provide recommended extensions or settings for Python analysis, Ruff, Markdown, and config-file diagnostics

#### Scenario: Understanding LSP reuse
- **WHEN** a contributor asks whether Copilot, Claude Code, or Codex require separate LSP servers
- **THEN** the repository guidance SHALL explain that LSP is a shared protocol and the main question is which client/editor attaches to the chosen language server

### Requirement: AI tooling guidance is explicit about trade-offs
The repository SHALL document how Copilot, Claude, Codex, subagents, review flows, and optional MCP usage should be combined for this codebase.

#### Scenario: Choosing automation depth
- **WHEN** a contributor prepares a broad cleanup or implementation run
- **THEN** the guidance SHALL recommend when to use review-oriented passes, when to use subagents, and when to avoid unnecessary MCP or high-cost model usage
