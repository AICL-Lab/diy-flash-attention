# AI Agent Instructions: Spec-Driven Development (SDD)

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementation must use the specification documents in the `/specs` directory as the Single Source of Truth.

## Directory Context

### Specification Documents (`/specs/`)

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product feature definitions and acceptance criteria (PRD) |
| `/specs/rfc/` | Technical design documents and architecture RFCs |
| `/specs/api/` | API interface specifications (OpenAPI, etc.) |
| `/specs/db/` | Database schema specifications (if applicable) |
| `/specs/testing/` | BDD testing specifications (Gherkin feature files) |

### Documentation (`/docs/`)

| Directory | Purpose |
|-----------|---------|
| `/docs/en/` | English user documentation |
| `/docs/zh/` | Chinese user documentation |
| `/docs/.vitepress/` | VitePress configuration |

### Source Code

| Directory | Purpose |
|-----------|---------|
| `/kernels/` | GPU kernel implementations (matmul.py, flash_attn.py, modern_features.py) |
| `/utils/` | Utility modules (benchmark.py, validation.py, gpu_detect.py) |
| `/benchmarks/` | Performance benchmark scripts |
| `/tests/` | Unit tests and property-based tests |
| `/examples/` | Example code for users |

## AI Agent Workflow Instructions

When you (AI) are asked to develop a new feature, modify existing features, or fix bugs, **you MUST strictly follow this workflow. Do NOT skip any steps**:

### Step 1: Review Specifications

**MANDATORY FIRST STEP**

- Read relevant documents in `/specs` directory:
  - Product requirements in `/specs/product/`
  - Technical RFCs in `/specs/rfc/`
  - API specifications in `/specs/api/`
  - Testing specs in `/specs/testing/`
- **If the user's request conflicts with existing specs**:
  - STOP coding immediately
  - Point out the conflict
  - Ask the user whether to update specs first

**Example**:
```
User: "Add FP8 support to matmul"
AI: "Let me check the specs first... I see Requirement 8.2 in specs/product/flash-attention-prd.md mentions FP8 support. 
     The RFC 0001 also describes the architecture. I'll implement according to these specs."
```

### Step 2: Spec-First Update

**SPEC BEFORE CODE**

- If this is a new feature, or if existing interfaces/database structures need changes:
  - **MUST FIRST propose modifications or creation of corresponding Spec documents**
  - Wait for user confirmation of spec changes
  - Only then proceed to code implementation
- If specs already exist and are complete:
  - Confirm you've read and understood them
  - Proceed to implementation

**Example**:
```
AI: "I've created specs/product/fp8-support-prd.md with the following requirements:
     1. FP8 E4M3 format support
     2. FP8 E5M2 format support
     3. Automatic fallback to FP16
     
     Please confirm these specs before I start implementing."
```

### Step 3: Code Implementation

**100% SPEC COMPLIANCE**

When writing code, you MUST:
- 100% follow spec definitions (including variable names, API paths, data types, status codes)
- NOT add features not defined in specs (No Gold-Plating)
- Follow error handling exactly as specified
- Use exact interfaces defined in specs

**Prohibited**:
- ❌ Adding "nice-to-have" features not in specs
- ❌ Changing API endpoints without updating `/specs/api/`
- ❌ Modifying data structures without updating `/specs/db/` or RFCs
- ❌ Inventing new design patterns not discussed in RFCs

**Allowed**:
- ✅ Fixing bugs that contradict specs (but note the discrepancy)
- ✅ Optimizing implementation while keeping interfaces identical to specs
- ✅ Adding comments explaining spec requirements

### Step 4: Test Against Spec

**VALIDATE AGAINST ACCEPTANCE CRITERIA**

- Write unit and integration tests based on acceptance criteria in `/specs`
- Ensure test cases cover all boundary conditions described in specs
- For property-based tests, validate the properties defined in RFCs
- For BDD tests, implement scenarios from `/specs/testing/`

**Test Coverage Requirements**:
- kernels/: 85%+
- utils/: 90%+
- All acceptance criteria from specs must have corresponding tests

## Code Generation Rules

### Rule 1: API Changes
Any externally exposed API changes MUST:
1. First update `/specs/api/` (if it exists) or create API spec
2. Update `/specs/rfc/` with design changes
3. Get user confirmation
4. Then implement code changes

### Rule 2: Architecture Changes
Any architectural changes MUST:
1. Create new RFC in `/specs/rfc/` with sequential numbering (e.g., `0002-new-feature.md`)
2. Update existing RFCs if needed
3. Get user confirmation
4. Then implement

### Rule 3: Implementation Details
When implementing:
- Follow existing code style and conventions in the project
- Read surrounding code to understand local patterns
- Use same naming conventions
- Add comments explaining WHY (not WHAT) for complex logic
- Reference spec sections in comments when implementing spec requirements

**Example**:
```python
def flash_attention(q, k, v, causal=False, seq_lens=None):
    """
    FlashAttention forward pass.
    
    Implements Requirement 4.2 from specs/product/flash-attention-prd.md:
    "When computing attention, the Triton_Kernel shall use online softmax..."
    """
    # Step 1: Load Q block into SRAM (per RFC 0001 algorithm)
    ...
```

### Rule 4: Uncertainty Handling
If you encounter uncertain technical details:
1. Check `/specs/rfc/` for architecture decisions
2. If not covered, ASK the user - do NOT invent designs
3. Suggest updating specs before proceeding

## Conflict Resolution

When there's a conflict between:
- **User request vs Specs**: Point out conflict, ask for clarification
- **Specs vs Existing code**: Follow specs, fix code to match specs (but note the discrepancy)
- **Multiple specs conflict**: Ask user to resolve
- **Specs are outdated**: Suggest updating specs first

## Examples of Correct Workflow

### Example 1: New Feature Request

```
User: "Add support for Multi-Query Attention (MQA)"

AI Correct Response:
1. "Let me check existing specs..."
2. "I don't see MQA in current specs. I'll create:
   - specs/product/mqa-support-prd.md with requirements
   - specs/rfc/0002-mqa-architecture.md with design
3. Here's the draft... Please confirm."
4. [After confirmation] "Now I'll implement according to specs..."
5. [Implementation following spec exactly]
6. "I've added tests validating all acceptance criteria from the spec."
```

### Example 2: Bug Fix

```
User: "FlashAttention fails with head_dim=128"

AI Correct Response:
1. "Let me check the spec..."
2. "Per Requirement 4.7 in specs/product/flash-attention-prd.md:
   'THE Triton_Kernel SHALL support head_dim of 32 and 64'
   head_dim=128 is not currently supported by design."
3. "Options:
   a. Fix the error message to be clearer (spec-compliant)
   b. Add head_dim=128 support (requires spec update)
   Which would you prefer?"
```

### Example 3: Refactoring

```
User: "Refactor the matmul kernel"

AI Correct Response:
1. "Let me review the current spec..."
2. "RFC 0001 defines the kernel interface and autotune configs."
3. "I'll ensure the refactoring maintains spec-compliant interfaces."
4. "All existing tests will pass, confirming spec compliance."
```

## Important Reminders

⚠️ **NEVER**:
- Skip reading specs before coding
- Add features not in specs
- Change specs without user confirmation
- Invent designs not discussed in RFCs

✅ **ALWAYS**:
- Start with specs
- Follow specs exactly
- Validate against specs
- Update specs before code when needed
- Reference specs in commit messages and comments

---

**This document is the ground truth for AI agent behavior in this project.**
