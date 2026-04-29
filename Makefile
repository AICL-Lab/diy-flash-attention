.PHONY: install install-dev test test-cpu test-gpu lint format typecheck docs hooks-install hooks-run validate-openspec bench-matmul bench-flash demo clean help

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
	@echo "  Test & Quality:"
	@echo "    make test         - Run the default CPU-safe test path"
	@echo "    make test-cpu     - Run tests that work without CUDA kernels"
	@echo "    make test-gpu     - Run the full GPU test suite"
	@echo "    make lint         - Lint code with ruff"
	@echo "    make format       - Format code with ruff"
	@echo "    make typecheck    - Type check with mypy"
	@echo "    make docs         - Build the VitePress docs site"
	@echo "    make validate-openspec - Validate main specs and active changes"
	@echo "    make hooks-install - Install repo-local git hooks"
	@echo "    make hooks-run    - Run pre-commit across all files"
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
	pip install .

# Install in development mode
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	$(MAKE) test-cpu

test-cpu:
	pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py

test-gpu:
	pytest tests/ -v

# Lint code with ruff
lint:
	ruff check kernels/ utils/ tests/ benchmarks/ examples/

# Format code with ruff
format:
	ruff format kernels/ utils/ tests/ benchmarks/ examples/

# Type check with mypy
typecheck:
	mypy kernels/ utils/

docs:
	npm run docs:build

validate-openspec:
	openspec validate --specs --json || echo "No active changes or specs valid"

hooks-install:
	pre-commit install --hook-type pre-commit

hooks-run:
	pre-commit run --all-files

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
