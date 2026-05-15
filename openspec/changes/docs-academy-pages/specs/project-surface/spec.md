## ADDED Requirements

### Requirement: English and Chinese GitHub Pages act as a learning academy entrypoint
The English and Chinese GitHub Pages surfaces SHALL provide aligned structured entrypoints for learning path, paper guide, and concept map content that help readers navigate the repository's educational FlashAttention scope.

#### Scenario: First-time visitor on GitHub Pages
- **WHEN** a reader lands on the English or Chinese docs homepage
- **THEN** the site SHALL present the same guided entrypoints for where to start learning, what papers to read, and which concepts connect to which repository documents

#### Scenario: Reader switches between English and Chinese homepages
- **WHEN** a reader compares the English and Chinese docs homepages
- **THEN** both surfaces SHALL expose the same academy portal structure and equivalent entrypoints, with locale-appropriate copy only

### Requirement: Public SVG assets remain legible across light and dark themes
The GitHub Pages site SHALL ensure primary SVG assets used in the docs surface remain readable in both light and dark themes.

#### Scenario: Reader switches site theme
- **WHEN** a reader views the docs surface in light mode and dark mode
- **THEN** primary SVG assets on the public docs surface SHALL remain legible without losing essential labels, contrast, or structure
