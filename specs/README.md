# Specifications

This directory contains all specification documents for the DIY FlashAttention project. These specs serve as the **Single Source of Truth** for all development work.

## Directory Structure

```
specs/
├── product/              # Product requirements and feature definitions
│   └── flash-attention-prd.md    # Product Requirements Document (PRD)
├── rfc/                  # Technical design documents and architecture RFCs
│   └── 0001-core-architecture.md # Core system architecture design
├── api/                  # API interface specifications (OpenAPI, etc.)
├── db/                   # Database schema specifications (if applicable)
└── testing/              # BDD testing specifications
    └── flash-attention.feature   # Behavior-driven test scenarios
```

## Specification Types

### Product Requirements (`/product/`)

- **Purpose**: Define what the product should do from a user/business perspective
- **Format**: User stories with acceptance criteria
- **Audience**: Product managers, developers, stakeholders
- **Example**: `flash-attention-prd.md` - Contains 11 requirements covering kernel implementation, benchmarking, validation, documentation, and collaboration standards

### RFCs - Request for Comments (`/rfc/`)

- **Purpose**: Document technical design decisions and architecture
- **Format**: Structured design documents with context, architecture, components, algorithms
- **Audience**: Developers, architects, reviewers
- **Example**: `0001-core-architecture.md` - Core system architecture including GPU memory hierarchy, kernel designs, data models, and error handling
- **Naming Convention**: Sequential numbering (0001, 0002, ...) for easy reference

### API Specifications (`/api/`)

- **Purpose**: Define external interfaces and contracts
- **Format**: OpenAPI YAML, GraphQL schemas, or human-readable specs
- **Audience**: API consumers, developers
- **Status**: Currently defined inline in RFC documents

### Database Specifications (`/db/`)

- **Purpose**: Define data models and schema
- **Format**: DBML, SQL, or ER diagrams
- **Audience**: Database administrators, backend developers
- **Status**: Not applicable for this project (no database)

### Testing Specifications (`/testing/`)

- **Purpose**: Define behavior-driven test scenarios
- **Format**: Gherkin feature files (Given-When-Then)
- **Audience**: QA engineers, developers
- **Example**: `flash-attention.feature` - BDD scenarios for all major features

## How to Use Specs

### For Developers

1. **Before starting work**: Read relevant specs to understand requirements
2. **When implementing**: Follow component interfaces and error handling defined in specs
3. **When unsure**: Check specs before making design decisions
4. **When proposing changes**: Update specs first, then implement

### For Reviewers

1. **Code reviews**: Verify implementation matches spec definitions
2. **Acceptance criteria**: Check all acceptance criteria are met
3. **Property tests**: Ensure property-based tests validate spec properties

### For Contributors

1. **New features**: Start by creating/updating specs in `/specs/product/`
2. **Technical designs**: Write RFC in `/specs/rfc/` with sequential numbering
3. **API changes**: Update `/specs/api/` with new interface definitions
4. **Testing**: Add BDD scenarios to `/specs/testing/`

## Spec-Driven Development Workflow

This project follows **Spec-Driven Development (SDD)**:

1. **Spec First**: Requirements and designs are written before implementation
2. **Implementation**: Code is written to match specs exactly
3. **Validation**: Tests verify implementation against specs
4. **Sync**: Specs and code are updated together in same PR

See `AGENTS.md` for detailed AI agent workflow instructions.

## Traceability

Requirements are traced throughout the project:

```
PRD Requirements → RFC Design → Implementation → Tests → Validation
```

Example trace:
- `Requirement 4.1` (PRD) → `Section 2: FlashAttention Kernel` (RFC) → `kernels/flash_attn.py` → `tests/test_flash.py` → `specs/testing/flash-attention.feature`

## Versioning

Specs follow project versioning:
- Major changes: New RFC for architectural changes
- Minor changes: Updates to existing requirements or designs
- All spec changes should be committed with clear messages referencing affected requirements
