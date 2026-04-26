## MODIFIED Requirements

### Requirement: Quality gates stay high-signal
The repository SHALL keep only automation that materially protects the current Python + Triton + VitePress codebase, and the CI workflow SHALL mirror the documented local validation commands without redundant triggers or duplicate checks.

#### Scenario: Running continuous integration
- **WHEN** CI executes for repository changes
- **THEN** it SHALL validate OpenSpec structure, docs buildability, code style, type safety, and CPU-safe tests without adding redundant workflows

#### Scenario: Narrowing workflow triggers
- **WHEN** a workflow is configured or edited
- **THEN** its trigger paths and jobs SHALL reflect the files and validation scope it actually protects instead of firing on unrelated repository changes

### Requirement: Editor and LSP guidance is shared
The repository SHALL commit workspace guidance for Python, Markdown, TOML, YAML, and JSON editing so contributors get consistent diagnostics, and that guidance SHALL stay machine-agnostic.

#### Scenario: Configuring a supported editor
- **WHEN** a contributor opens the repository in an editor that supports workspace recommendations
- **THEN** the repository SHALL provide recommended extensions or settings for Python analysis, Ruff, Markdown, and config-file diagnostics

#### Scenario: Understanding LSP reuse
- **WHEN** a contributor asks whether Copilot, Claude Code, or Codex require separate LSP servers
- **THEN** the repository guidance SHALL explain that LSP is a shared protocol and the main question is which client/editor attaches to the chosen language server

### Requirement: AI tooling guidance is explicit about trade-offs
The repository SHALL document how Copilot, Claude, Codex/Opencode-style tools, `gh`, review flows, subagents, CLI skills, and optional MCP usage should be combined for this codebase, and that guidance SHALL stay specific to the repository's actual maintenance needs.

#### Scenario: Choosing automation depth
- **WHEN** a contributor prepares a broad cleanup or implementation run
- **THEN** the guidance SHALL recommend when to use review-oriented passes, when to use subagents, and when to avoid unnecessary MCP or high-cost model usage

#### Scenario: Reviewing committed AI instructions
- **WHEN** project AI instruction files are added or updated
- **THEN** they SHALL describe real repository workflows and available tools instead of embedding unrelated external protocols or generic boilerplate

## ADDED Requirements

### Requirement: Dependency and version policy stays coherent
The repository SHALL keep Python, Node, package, and documentation version requirements coherent across configuration files, CI, and contributor-facing guidance.

#### Scenario: Updating toolchain baselines
- **WHEN** the repository raises, lowers, or clarifies a supported Python, Node, CUDA, Triton, or dependency baseline
- **THEN** configuration files, CI setup, and public docs SHALL be updated together

#### Scenario: Evaluating a new dependency
- **WHEN** a dependency or tool is proposed for addition
- **THEN** the repository SHALL prefer the smallest change that materially reduces maintenance pain and reject additions that only add optional complexity
