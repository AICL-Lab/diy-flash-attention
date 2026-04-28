"""Tests for BlockMask and attention mask DSL."""

from __future__ import annotations

import pytest
import torch

from kernels import BlockMask, compose_block_masks, create_block_mask


class TestBlockMaskCreation:
    """Test BlockMask dataclass creation."""

    def test_basic_creation(self):
        """Test BlockMask creation."""
        mask = BlockMask(
            mask_matrix=torch.ones(8, 8, dtype=torch.bool),
            query_block_size=32,
            key_block_size=32,
            mask_type="full",
        )
        assert mask.query_block_size == 32
        assert mask.key_block_size == 32
        assert mask.mask_type == "full"
        assert mask.get_mask_matrix().shape == (8, 8)

    def test_get_mask_matrix(self):
        """Test mask matrix retrieval."""
        matrix = torch.ones(4, 4, dtype=torch.bool)
        mask = BlockMask(
            mask_matrix=matrix,
            query_block_size=64,
            key_block_size=64,
            mask_type="full",
        )
        result = mask.get_mask_matrix()
        assert torch.equal(result, matrix)


class TestCreateBlockMask:
    """Test block mask factory functions."""

    def test_causal_pattern(self):
        """Test causal block mask generation."""
        mask = create_block_mask(
            pattern="causal",
            query_len=256,
            key_len=256,
            block_size=32,
        )

        # Verify lower triangular structure
        matrix = mask.get_mask_matrix()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i >= j:
                    assert matrix[i, j] == 1, f"Block ({i}, {j}) should be 1 (causal)"
                else:
                    assert matrix[i, j] == 0, f"Block ({i}, {j}) should be 0 (causal)"

    def test_full_pattern(self):
        """Test full attention block mask."""
        mask = create_block_mask(
            pattern="full",
            query_len=256,
            key_len=256,
            block_size=32,
        )

        matrix = mask.get_mask_matrix()
        assert torch.all(matrix == 1)

    def test_sliding_window_pattern(self):
        """Test sliding window block mask."""
        window_size = 128
        mask = create_block_mask(
            pattern="sliding_window",
            query_len=256,
            key_len=256,
            block_size=32,
            sliding_window=window_size,
        )

        matrix = mask.get_mask_matrix()
        # Check that mask is banded around diagonal
        assert matrix.sum() > 0  # Has some ones
        assert matrix.sum() < matrix.numel()  # Not all ones

    def test_prefix_lm_pattern(self):
        """Test prefix LM block mask."""
        prefix_len = 64
        mask = create_block_mask(
            pattern="prefix_lm",
            query_len=256,
            key_len=256,
            block_size=32,
            prefix_len=prefix_len,
        )

        matrix = mask.get_mask_matrix()
        # First prefix blocks should be fully attended
        prefix_blocks = (prefix_len + 31) // 32
        assert torch.all(matrix[:, :prefix_blocks] == 1)

    def test_block_size_calculation(self):
        """Test that block sizes are calculated correctly."""
        mask = create_block_mask(
            pattern="full",
            query_len=256,
            key_len=256,
            block_size=64,
        )

        # 256 / 64 = 4 blocks per dimension
        assert mask.mask_matrix.shape == (4, 4)

    def test_non_square_mask(self):
        """Test mask with different query and key lengths."""
        mask = create_block_mask(
            pattern="full",
            query_len=128,
            key_len=256,
            block_size=32,
        )

        # 128 / 32 = 4, 256 / 32 = 8
        assert mask.mask_matrix.shape == (4, 8)


class TestComposeBlockMasks:
    """Test mask composition operations."""

    def test_intersect(self):
        """Test mask intersection (AND operation)."""
        causal = create_block_mask("causal", 256, 256, 32)
        full = create_block_mask("full", 256, 256, 32)

        result = compose_block_masks(causal, full, operation="intersect")

        # Intersection of causal and full should be causal
        assert torch.equal(result.mask_matrix, causal.mask_matrix)

    def test_union(self):
        """Test mask union (OR operation)."""
        # Create two masks with different patterns
        mask1 = BlockMask(
            mask_matrix=torch.tensor([[1, 0], [0, 0]], dtype=torch.bool),
            query_block_size=32,
            key_block_size=32,
            mask_type="custom",
        )
        mask2 = BlockMask(
            mask_matrix=torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
            query_block_size=32,
            key_block_size=32,
            mask_type="custom",
        )

        result = compose_block_masks(mask1, mask2, operation="union")

        expected = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
        assert torch.equal(result.mask_matrix, expected)

    def test_composed_mask_type(self):
        """Test that composed mask has correct type."""
        mask1 = create_block_mask("causal", 256, 256, 32)
        mask2 = create_block_mask("full", 256, 256, 32)

        result = compose_block_masks(mask1, mask2, operation="intersect")
        assert result.mask_type == "composed"


class TestBlockMaskApply:
    """Test applying BlockMask to attention scores."""

    def test_apply_to_scores_shape(self):
        """Test that apply_to_scores preserves shape."""
        mask = create_block_mask("causal", 128, 128, 32)

        scores = torch.zeros(1, 4, 128, 128)  # (batch, heads, q_len, k_len)
        masked = mask.apply_to_scores(scores)

        assert masked.shape == scores.shape


class TestInvalidPatterns:
    """Test error handling for invalid patterns."""

    def test_unknown_pattern_raises(self):
        """Test that unknown pattern raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            create_block_mask(
                pattern="invalid_pattern",
                query_len=256,
                key_len=256,
                block_size=32,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
