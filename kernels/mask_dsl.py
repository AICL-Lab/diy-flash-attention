"""
BlockMask and Attention Mask DSL

This module provides a block-level mask abstraction for flexible attention patterns.
Educational focus: teach students how attention masks work at the block level,
which is fundamental to FlashAttention's memory efficiency.

Key concepts:
- Block-level sparsity: Mask operates on blocks, not individual tokens
- Pattern composition: Combine masks with intersection/union
- Memory efficiency: Block masks are much smaller than token-level masks

Supported patterns:
- causal: Lower triangular (autoregressive models)
- full: All positions attend to all positions
- sliding_window: Local attention within a window
- prefix_lm: Prefix fully attended, rest causal (encoder-decoder)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BlockMask:
    """
    Block-level attention mask abstraction.

    Instead of a full (seq_len, seq_len) token-level mask, we use a
    (n_query_blocks, n_key_blocks) block-level mask. This is more memory
    efficient and aligns with FlashAttention's tiled computation.

    Attributes:
        mask_matrix: Boolean tensor of shape (n_query_blocks, n_key_blocks)
        query_block_size: Number of query tokens per block
        key_block_size: Number of key tokens per block
        mask_type: Pattern name for debugging/logging
    """

    mask_matrix: torch.Tensor
    query_block_size: int
    key_block_size: int
    mask_type: str

    def get_mask_matrix(self) -> torch.Tensor:
        """Return boolean mask matrix."""
        return self.mask_matrix.bool()

    def apply_to_scores(
        self,
        scores: torch.Tensor,
        score_mod: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Apply block mask to attention scores.

        Expands the block mask to token level and masks out positions.

        Args:
            scores: Attention scores of shape (batch, heads, query_len, key_len)
            score_mod: Optional score modification function (deferred to Phase B)

        Returns:
            Masked scores with -inf for masked positions
        """
        batch, heads, query_len, key_len = scores.shape

        n_query_blocks = self.mask_matrix.shape[0]
        n_key_blocks = self.mask_matrix.shape[1]

        # Calculate actual lengths from blocks
        actual_query_len = n_query_blocks * self.query_block_size
        actual_key_len = n_key_blocks * self.key_block_size

        # Handle case where scores is smaller than mask
        query_len_to_use = min(query_len, actual_query_len)
        key_len_to_use = min(key_len, actual_key_len)

        # Expand block mask to token level
        token_mask = torch.zeros(
            query_len_to_use, key_len_to_use, dtype=torch.bool, device=scores.device
        )

        for i in range(n_query_blocks):
            for j in range(n_key_blocks):
                q_start = i * self.query_block_size
                q_end = min(q_start + self.query_block_size, query_len_to_use)
                k_start = j * self.key_block_size
                k_end = min(k_start + self.key_block_size, key_len_to_use)

                if self.mask_matrix[i, j]:
                    token_mask[q_start:q_end, k_start:k_end] = 1

        # Pad token_mask to match scores shape if needed
        if query_len_to_use < query_len or key_len_to_use < key_len:
            full_mask = torch.zeros(query_len, key_len, dtype=torch.bool, device=scores.device)
            full_mask[:query_len_to_use, :key_len_to_use] = token_mask
            token_mask = full_mask

        # Apply to scores: broadcast over batch and heads
        masked_scores = scores.clone()
        masked_scores = torch.where(
            token_mask[None, None, :, :],
            scores,
            torch.full_like(scores, float("-inf")),
        )

        if score_mod is not None:
            # Apply score modification (deferred to Phase B)
            # This would allow custom modifications like relative position bias
            pass

        return masked_scores


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
        pattern: Mask pattern name
            - "causal": Lower triangular (autoregressive)
            - "full": All positions attend to all positions
            - "sliding_window": Local attention within window
            - "prefix_lm": Prefix fully attended, rest causal
        query_len: Total query sequence length
        key_len: Total key sequence length
        block_size: Number of tokens per block (default: 128)
        causal: Alias for pattern="causal" (for convenience)
        sliding_window: Window size for sliding_window pattern
        prefix_len: Prefix length for prefix_lm pattern

    Returns:
        BlockMask with the specified pattern

    Raises:
        ValueError: If pattern is unknown or required parameters are missing
    """
    # Handle convenience alias
    if causal:
        pattern = "causal"

    n_query_blocks = (query_len + block_size - 1) // block_size
    n_key_blocks = (key_len + block_size - 1) // block_size

    if pattern == "causal":
        # Lower triangular: query block i can attend to key block j iff j <= i
        mask = torch.tril(torch.ones(n_query_blocks, n_key_blocks, dtype=torch.bool))

    elif pattern == "full":
        # All blocks: every query attends to every key
        mask = torch.ones(n_query_blocks, n_key_blocks, dtype=torch.bool)

    elif pattern == "sliding_window":
        # Diagonal band: only attend within window
        if sliding_window is None:
            raise ValueError("sliding_window pattern requires sliding_window parameter")

        mask = torch.zeros(n_query_blocks, n_key_blocks, dtype=torch.bool)
        window_blocks = (sliding_window + block_size - 1) // block_size

        for i in range(n_query_blocks):
            # Allow attention to keys within window
            j_start = max(0, i - window_blocks)
            j_end = min(n_key_blocks, i + window_blocks + 1)
            mask[i, j_start:j_end] = 1

    elif pattern == "prefix_lm":
        # Prefix fully attended, rest causal
        # Used in encoder-decoder models where prefix is the encoder output
        if prefix_len is None:
            raise ValueError("prefix_lm pattern requires prefix_len parameter")

        mask = torch.zeros(n_query_blocks, n_key_blocks, dtype=torch.bool)
        prefix_blocks = (prefix_len + block_size - 1) // block_size

        # All queries can attend to prefix keys
        mask[:, :prefix_blocks] = 1

        # Causal for the rest
        for i in range(n_query_blocks):
            for j in range(prefix_blocks, n_key_blocks):
                if j <= i:
                    mask[i, j] = 1

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return BlockMask(
        mask_matrix=mask,
        query_block_size=block_size,
        key_block_size=block_size,
        mask_type=pattern,
    )


def compose_block_masks(
    mask1: BlockMask,
    mask2: BlockMask,
    operation: str = "intersect",
) -> BlockMask:
    """
    Compose two block masks.

    Args:
        mask1: First mask
        mask2: Second mask (must have same shape as mask1)
        operation: Composition operation
            - "intersect": AND (only positions allowed by both)
            - "union": OR (positions allowed by either)

    Returns:
        New BlockMask with composed pattern

    Raises:
        ValueError: If masks have different shapes or operation is unknown
    """
    if mask1.mask_matrix.shape != mask2.mask_matrix.shape:
        raise ValueError(
            f"Mask shapes must match: {mask1.mask_matrix.shape} vs {mask2.mask_matrix.shape}"
        )

    if operation == "intersect":
        combined = mask1.mask_matrix & mask2.mask_matrix
    elif operation == "union":
        combined = mask1.mask_matrix | mask2.mask_matrix
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return BlockMask(
        mask_matrix=combined,
        query_block_size=mask1.query_block_size,
        key_block_size=mask1.key_block_size,
        mask_type="composed",
    )


if __name__ == "__main__":
    print("Testing BlockMask patterns...")
    print("=" * 60)

    # Test causal
    causal = create_block_mask("causal", 256, 256, 32)
    print("\nCausal mask (8x8 blocks):")
    print(causal.mask_matrix.int())

    # Test full
    full = create_block_mask("full", 256, 256, 32)
    print("\nFull mask (8x8 blocks):")
    print(f"  All ones: {torch.all(full.mask_matrix == 1).item()}")

    # Test sliding window
    sliding = create_block_mask("sliding_window", 256, 256, 32, sliding_window=64)
    print("\nSliding window mask (window=64):")
    print(sliding.mask_matrix.int())

    # Test prefix LM
    prefix = create_block_mask("prefix_lm", 256, 256, 32, prefix_len=64)
    print("\nPrefix LM mask (prefix=64):")
    print(prefix.mask_matrix.int())

    # Test composition
    print("\n" + "=" * 60)
    print("Testing mask composition...")

    composed = compose_block_masks(causal, full, operation="intersect")
    print(
        f"Intersect causal + full = causal: {torch.equal(composed.mask_matrix, causal.mask_matrix)}"
    )

    print("\n✓ All BlockMask tests passed")
