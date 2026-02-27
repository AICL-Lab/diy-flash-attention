"""Pytest configuration and fixtures for DIY FlashAttention tests."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available):
    """Get the device to use for tests."""
    if cuda_available:
        return torch.device("cuda")
    pytest.skip("CUDA not available")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


@pytest.fixture
def small_matrices(device):
    """Create small test matrices."""
    M, N, K = 64, 64, 64
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    return a, b


@pytest.fixture
def medium_matrices(device):
    """Create medium test matrices."""
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    return a, b


@pytest.fixture
def attention_tensors(device):
    """Create test tensors for attention."""
    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    q = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch.float16)
    return q, k, v
