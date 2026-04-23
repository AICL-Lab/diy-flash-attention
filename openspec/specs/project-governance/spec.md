## Purpose

Define the repository's OpenSpec-first governance model, AI-assistant operating rules, review discipline, and archive-ready scope control.

## Requirements

### Requirement: OpenSpec is the mandatory change entry point
The repository SHALL use OpenSpec as the mandatory workflow for scoped project changes.

#### Scenario: Starting non-trivial work
- **WHEN** a contributor starts a non-trivial change
- **THEN** the contributor SHALL begin with `explore` or `propose`, establish or select an active change, and avoid direct implementation without an OpenSpec task trail

#### Scenario: Finishing a change
- **WHEN** all tasks in an active change are complete
- **THEN** the repository workflow SHALL treat the change as ready for archive instead of leaving stale active changes behind

### Requirement: Governance docs are repository-specific
The repository SHALL keep project instructions in root-level guidance files that describe this repository's Triton, OpenSpec, docs, and review workflow rather than generic AI boilerplate.

#### Scenario: Reading agent instructions
- **WHEN** a human or AI assistant reads `AGENTS.md`, `CLAUDE.md`, or Copilot instructions
- **THEN** the guidance SHALL explain repository-specific constraints, preferred workflows, and quality expectations for this project

### Requirement: Review and branch discipline stay lightweight
The repository SHALL prefer short-lived branches, prompt review, and fast reintegration over long-lived divergence between local and cloud work.

#### Scenario: Preparing a non-trivial merge
- **WHEN** a contributor finishes a non-trivial implementation slice
- **THEN** the workflow SHALL require a review pass such as `/review` before merge or archival handoff

#### Scenario: Managing work in progress
- **WHEN** multiple ideas compete for attention
- **THEN** the workflow SHALL favor one active change at a time for repository-wide cleanup work rather than scattering partially merged branches

### Requirement: Archive-ready scope control
The repository SHALL bias toward finishing, clarifying, and de-risking the existing educational scope instead of expanding feature scope.

#### Scenario: Evaluating a proposed enhancement
- **WHEN** a change introduces new complexity that is not necessary for the educational FlashAttention scope
- **THEN** the workflow SHALL prefer documenting the idea for later rather than expanding the current repository surface
