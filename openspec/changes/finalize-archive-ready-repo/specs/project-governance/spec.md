## MODIFIED Requirements

### Requirement: OpenSpec is the mandatory change entry point
The repository SHALL use OpenSpec as the mandatory workflow for scoped project changes, and broad cleanup work SHALL not proceed without an explicit active change even if an earlier stabilization change has already completed.

#### Scenario: Starting non-trivial work
- **WHEN** a contributor starts a non-trivial change
- **THEN** the contributor SHALL begin with `explore` or `propose`, establish or select an active change, and avoid direct implementation without an OpenSpec task trail

#### Scenario: Restarting broad cleanup after a completed change
- **WHEN** repository-wide cleanup resumes after the current active change has been completed
- **THEN** the contributor SHALL create or select a new active change instead of editing directly against completed tasks

#### Scenario: Finishing a change
- **WHEN** all tasks in an active change are complete
- **THEN** the repository workflow SHALL treat the change as ready for archive instead of leaving stale active changes behind

### Requirement: Governance docs are repository-specific
The repository SHALL keep project instructions in root-level guidance files that describe this repository's Triton, OpenSpec, docs, review workflow, and actual tool availability rather than generic AI boilerplate or external process overlays.

#### Scenario: Reading agent instructions
- **WHEN** a human or AI assistant reads `AGENTS.md`, `CLAUDE.md`, or Copilot instructions
- **THEN** the guidance SHALL explain repository-specific constraints, preferred workflows, and quality expectations for this project

#### Scenario: Auditing instruction files
- **WHEN** committed guidance files are reviewed during cleanup
- **THEN** content that refers to unavailable tools, unrelated protocols, or duplicated generic instructions SHALL be removed or consolidated

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

## ADDED Requirements

### Requirement: Handoff state remains explicit
The repository SHALL leave an explicit handoff state for broad cleanup work so the next contributor or agent can see what is done, what remains, and which questions are intentionally deferred.

#### Scenario: Transitioning between contributors or agents
- **WHEN** a broad cleanup phase pauses or changes hands
- **THEN** the active change tasks and adjacent governance notes SHALL identify completed work, remaining tasks, and unresolved questions without forcing the next contributor to rediscover scope
