## ADDED Requirements

### Requirement: Educational kernel scope stays explicit
The repository SHALL keep the FlashAttention kernel surface explicitly educational and forward-only across specs, docs, and examples.

#### Scenario: Describing implemented scope
- **WHEN** a contributor updates repository documentation or specs for the kernel surface
- **THEN** the update SHALL describe the implemented FlashAttention path as forward-only unless the repository also adds and validates backward support
