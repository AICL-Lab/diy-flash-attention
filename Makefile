.PHONY: install test bench-matmul bench-flash demo clean help

# Default target
help:
	@echo "DIY FlashAttention - Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install      - Install dependencies"
	@echo "    make install-dev  - Install in development mode"
	@echo ""
	@echo "  Run:"
	@echo "    make demo         - Run quick start demo"
	@echo "    make experiment   - Run block size experiment"
	@echo "    make visualize    - Visualize tiling strategy"
	@echo ""
	@echo "  Benchmark:"
	@echo "    make bench-matmul - Run matrix multiplication benchmark"
	@echo "    make bench-flash  - Run FlashAttention benchmark"
	@echo "    make bench-all    - Run all benchmarks"
	@echo ""
	@echo "  Test:"
	@echo "    make test         - Run all tests"
	@echo "    make gpu-info     - Show GPU information"
	@echo ""
	@echo "    make clean        - Clean cache files"
	@echo ""
	@echo "  Advanced:"
	@echo "    make advanced     - Run advanced usage examples"
	@echo "    make report       - Generate benchmark report"
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

# Run block size experiment
experiment:
	python examples/block_size_experiment.py

# Visualize tiling strategy
visualize:
	python examples/visualize_tiling.py

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

# Run advanced usage examples
advanced:
	python examples/advanced_usage.py

# Generate benchmark report
report:
	python scripts/run_all_benchmarks.py --output benchmark_report.md
	@echo "Report saved to benchmark_report.md"

# Run all benchmarks and generate report
bench-report: bench-all report
