## ADDED Requirements

### Requirement: Root governance docs become authoritative
The repository SHALL keep `AGENTS.md`, `CLAUDE.md`, and Copilot instructions as the authoritative workflow entrypoints for AI-assisted development.

#### Scenario: Looking up project workflow
- **WHEN** a contributor or agent needs workflow guidance
- **THEN** the contributor or agent SHALL be able to use the root governance docs without relying on duplicate legacy guidance files

### Requirement: OpenSpec cleanup runs use one active change
The repository SHALL run broad cleanup work through a single active change until the repository returns to an archive-ready baseline.

#### Scenario: Continuing repository-wide cleanup
- **WHEN** a maintainer resumes the stabilization effort
- **THEN** the maintainer SHALL continue from the active stabilization change instead of opening parallel cleanup changes by default
