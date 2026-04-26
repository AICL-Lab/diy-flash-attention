## MODIFIED Requirements

### Requirement: README and GitHub Pages tell one story
The repository SHALL present the project as a hands-on educational FlashAttention/Triton implementation with one coherent positioning across README, bilingual docs entrypoints, and GitHub Pages.

#### Scenario: First-time visitor on GitHub
- **WHEN** a new visitor lands on the repository README
- **THEN** the page SHALL explain what the project teaches, what scope it covers, and where to continue in the docs site

#### Scenario: First-time visitor on GitHub Pages
- **WHEN** a new visitor lands on the docs homepage
- **THEN** the site SHALL act as a project showcase and learning entrypoint rather than duplicating README verbatim

#### Scenario: Switching between languages
- **WHEN** a reader moves between the English and Chinese public entrypoints
- **THEN** the repository SHALL keep equivalent navigation structure and consistent project positioning across both languages

### Requirement: GitHub repository metadata is aligned
The GitHub repository SHALL keep its description, homepage, and topics aligned with the README and GitHub Pages positioning, and those values SHALL only be updated after the public messaging they summarize is finalized.

#### Scenario: Repository About section
- **WHEN** a user views the repository About panel
- **THEN** the description and homepage SHALL reinforce the same Triton/FlashAttention learning narrative and include the GitHub Pages URL

#### Scenario: Refreshing About metadata
- **WHEN** project messaging changes across README or Pages
- **THEN** the repository SHALL update About metadata from that finalized messaging instead of inventing separate wording

### Requirement: Public docs avoid low-value duplication
The repository SHALL prefer concise, high-signal documentation over duplicated changelog mirrors, stale process docs, or generic filler pages, and it SHALL keep one clear authority for release/history tracking.

#### Scenario: Maintaining docs
- **WHEN** a document duplicates information already better maintained elsewhere
- **THEN** the repository SHALL consolidate or remove that document instead of preserving redundant content

#### Scenario: Maintaining changelog surfaces
- **WHEN** readers need project history or cleanup context
- **THEN** the repository SHALL route them to a single maintained changelog authority instead of keeping multiple parallel changelog narratives in README and docs
