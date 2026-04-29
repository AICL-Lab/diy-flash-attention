# Spec: attention-mask-dsl

## Overview

Implement `BlockMask` and mask composition helpers supporting causal, full, sliding_window, and prefix_lm attention patterns. Phase A establishes the abstraction foundation; Phase B expands patterns and optimization.

## Public API

```python
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

@dataclass
class BlockMask:
    """
    Block-level attention mask abstraction.
    
    Represents which (query_block, key_block) pairs should be computed.
    Enables efficient block-sparse attention without explicit masking on every token.
    """
    # Shape: (n_query_blocks, n_key_blocks)
    # False = skip this block (no attention between Q[i] and K[j])
    # True = compute attention for this block
    _mask_matrix: torch.Tensor
    
    # Metadata
    query_block_size: int
    key_block_size: int
    mask_type: str           # "causal", "full", "sliding_window", "prefix_lm", "custom"
    
    def get_mask_matrix(self) -> torch.Tensor:
        """Return boolean mask matrix (blocks to compute)."""
    
    def apply_to_scores(
        self,
        scores: torch.Tensor,      # (batch, heads, seq_len, seq_len)
        score_mod: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Apply block mask to attention scores.
        
        Args:
            scores: Attention scores (before softmax)
            score_mod: Optional per-score modification function
        
        Returns:
            Masked scores (masked positions set to -inf)
        """

def create_block_mask(
    pattern: str,
    query_len: int,
    key_len: int,
    block_size: int = 128,
    causal: bool = False,
    sliding_window: Optional[int] = None,
    prefix_len: Optional[int] = None,
) -> BlockMask:
    """
    Create a block-level attention mask.
    
    Args:
        pattern: "causal" | "full" | "sliding_window" | "prefix_lm"
        query_len, key_len: Attention dimensions
        block_size: Block size for sparse masking
        causal: Force causal if pattern is "full" (for hybrid masking)
        sliding_window: Window size for sliding window attention
        prefix_len: Prefix length for prefix LM pattern
    
    Returns:
        BlockMask object
    
    Raises:
        ValueError: Invalid pattern or inconsistent parameters
    """

def compose_block_masks(
    mask1: BlockMask,
    mask2: BlockMask,
    operation: str = "intersect",
) -> BlockMask:
    """
    Combine multiple block masks.
    
    Args:
        mask1, mask2: Block masks to combine
        operation: "intersect" (AND) | "union" (OR)
    
    Returns:
        Combined BlockMask
    """

def apply_block_mask_to_attention(
    q: torch.Tensor,           # (batch, seq_len, heads, head_dim)
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    score_mod: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Compute attention with block-level masking.
    
    This is the primary integration point with flash_attention kernels.
    
    Args:
        q, k, v: Attention tensors
        block_mask: BlockMask specification
        score_mod: Optional per-token score modification
    
    Returns:
        Attention output
    """
```

## Mask Patterns

### Causal Masking

- **Definition**: Query position can only attend to current and past key positions
- **Block pattern**: Lower triangular (or lower + diagonal)
- **Example**: Q[10] can attend to K[0..10], not K[11..seq_len]
- **Block representation**: Query block i can only attend to key block j if j <= i

### Full Masking

- **Definition**: All query positions attend to all key positions (no masking)
- **Block pattern**: Fully dense
- **Common for**: Prefill stages

### Sliding Window

- **Definition**: Query position attends to window around current position
- **Window size**: Fixed or relative
- **Example**: Q[100] can attend to K[80..120] if window_size=40
- **Phase A support**: Basic rectangular window (deferred: strided patterns)

### Prefix LM

- **Definition**: Causal within main sequence, but full attention to prefix
- **Pattern**: Prefix block fully attended, then causal for rest
- **Common for**: Prefix adaptation scenarios

## Behavior

### Block Mask Creation

1. **Pattern selection**: Choose mask type (causal, full, etc.)
2. **Block grid computation**: Divide seq_len into blocks of size `block_size`
3. **Mask matrix generation**: Boolean (n_blocks_q, n_blocks_k) matrix
4. **Storage optimization**: Sparse representation (block indices) for large matrices

### Score Modification

- Optional `score_mod` callback modifies attention scores per block
- Called after masking (masked positions stay -inf)
- Example: ALiBi position bias, relative position encoding

### Integration with Kernels

```python
# Within flash_attention_v2 or v1:
if block_mask is not None:
    # Skip masked blocks in kernel loop
    # Set masked scores to -inf
    # Integrate with online softmax

# Within persistent kernels:
# Use block mask to determine which (Q,K) blocks to load
```

## Supported Configurations

| Pattern | Query Len | Key Len | Block Size | Notes |
|---------|-----------|---------|-----------|-------|
| causal | 1-32768 | Same as Q | 32-512 | Standard SPD-only attention |
| full | 1-32768 | 1-32768 | 32-512 | Dense attention (prefill) |
| sliding_window | 1-32768 | 1-32768 | 32-512 | Window size configurable |
| prefix_lm | 1-32768 | 1-32768 | 32-512 | Prefix len configurable |

## Correctness Criteria

1. **Mask correctness**: Block mask produces same result as token-level masking
2. **Causal invariant**: Causal block mask == lower triangular, no forward attending
3. **Sliding window**: Query block i only attends to key blocks within window
4. **Prefix LM**: Prefix blocks fully attended, rest causal
5. **Composition**: `compose_block_masks(m1, m2, "intersect")` = intersection of masks
6. **Score modification**: score_mod callback applied without disturbing mask structure

## Integration Points

- **API entry**: `kernels/mask_dsl.py` (new module)
- **Kernel integration**: `flash_attn.py` and `flash_attn_v2.py` accept `block_mask` parameter
- **Benchmarks**: Mask pattern performance comparison (causal vs full vs sliding)
- **Tests**: `tests/test_mask_dsl.py` with correctness validation
- **Documentation**: Mask patterns guide with visual diagrams

## Testing Strategy

### Unit Tests (tests/test_mask_dsl.py)

```python
def test_create_block_mask_causal():
    """Causal block mask is lower triangular."""

def test_create_block_mask_full():
    """Full block mask is all True."""

def test_create_block_mask_sliding_window():
    """Sliding window mask has correct band structure."""

def test_create_block_mask_prefix_lm():
    """Prefix LM mask: prefix dense, rest causal."""

def test_apply_block_mask_to_scores():
    """Block mask applied to scores sets masked positions to -inf."""

def test_compose_block_masks_intersect():
    """Intersection of masks is correct."""

def test_block_mask_vs_token_level():
    """Block mask result == token-level masking result."""

def test_score_mod_callback():
    """Score modification applied without disrupting mask."""
```

### Integration Tests

- Attention with causal block mask == attention with causal token mask (numerical parity)
- Sliding window mask performance > causal (fewer computations)
- Block mask memory usage < full token-level mask storage

### Property-Based Tests

```python
@hypothesis.given(
    seq_len=..., block_size=..., pattern=...
)
def test_block_mask_properties(...):
    """Randomized correctness across parameters."""
```

## GPU Support

| Arch | Support | Notes |
|------|---------|-------|
| All | Full | Block mask is CPU-side logic; GPU integration transparent |

## Performance Expectations

| Pattern | Relative Speed | Notes |
|---------|----------------|-------|
| Full | 1.0x | Baseline (all blocks computed) |
| Causal | ~1.5x | ~50% fewer computations; exact depends on seq_len |
| Sliding window | Varies | Depends on window size relative to seq_len |
| Prefix LM | ~1.2x | Sparse gains smaller (prefix overhead) |

## Deferred (Phase B+)

- Custom mask_mod functions (arbitrary per-score modifications)
- Block-sparse execution optimization (skip computation in masked blocks)
- Ragged/irregular block sizes (adaptive block layout)
- Memory-efficient sparse mask storage (CRS/COO formats)
- Sliding window with strided patterns (ALiBi interactions)
- Composition with score_mod fusion (optimize combined operations)

## Success Criteria

- [ ] BlockMask data structure defined and tested
- [ ] `create_block_mask()` generates correct patterns for all 4 types
- [ ] Block mask application produces numerical parity with token-level masking
- [ ] Mask composition (intersect/union) works correctly
- [ ] Integration with flash_attention_v2 working
- [ ] All unit tests pass
- [ ] Benchmark shows expected speedups for sparse patterns
- [ ] Documentation includes visual mask pattern examples
