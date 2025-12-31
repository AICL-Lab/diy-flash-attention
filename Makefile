.PHONY: install test bench-matmul bench-flash demo clean help

# Default target
help:
	@echo "DIY FlashAttention - Available commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make demo         - Run quick start demo"
	@echo "  make bench-matmul - Run matrix multiplication benchmark"
	@echo "  make bench-flash  - Run FlashAttention benchmark"
	@echo "  make clean        - Clean cache files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Install in development mode
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run quick start demo
demo:
	python examples/quick_start.py

# Run matmul benchmark
bench-matmul:
	python benchmarks/bench_matmul.py

# Run matmul benchmark with block size testing
bench-matmul-blocks:
	python benchmarks/bench_matmul.py --test-block-sizes

# Run FlashAttention benchmark
bench-flash:
	python benchmarks/bench_flash.py

# Run FlashAttention benchmark with causal masking
bench-flash-causal:
	python benchmarks/bench_flash.py --causal

# Run all benchmarks
bench-all: bench-matmul bench-flash

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".triton" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Check GPU
gpu-info:
	python -c "from utils import detect_gpu, print_gpu_info; print_gpu_info(detect_gpu())"
