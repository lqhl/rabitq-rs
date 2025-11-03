.PHONY: help build-python install-python test-python clean-python bench-local bench-docker lint format

# NOTE: All builds automatically use optimizations from .cargo/config.toml:
# - target-cpu=native (enables AVX2/AVX-512 based on CPU capabilities)
# - llvm-args=--ffast-math (equivalent to C++ -ffast-math)

help:
	@echo "RaBitQ-RS Python Bindings - Available Commands"
	@echo ""
	@echo "Build & Install:"
	@echo "  make build-python      - Build Python package with AVX-512 (development mode)"
	@echo "  make build-python-compat - Build Python package without AVX-512 (compatibility)"
	@echo "  make install-python    - Install Python package from wheel (with AVX-512)"
	@echo "  make install-python-compat - Install Python package from wheel (without AVX-512)"
	@echo "  make clean-python      - Clean Python build artifacts"
	@echo ""
	@echo "Testing:"
	@echo "  make test-python       - Run Python integration tests"
	@echo "  make test-rust         - Run Rust tests with Python feature"
	@echo "  make test              - Run all tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench-local       - Run ann-benchmarks locally"
	@echo "  make bench-docker      - Run ann-benchmarks in Docker"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Run Rust linter"
	@echo "  make format            - Format Rust code"
	@echo ""
	@echo "NOTE: All builds use optimizations from .cargo/config.toml (target-cpu=native + fast-math)"
	@echo ""

# Build Python package in development mode
# Optimizations (target-cpu=native, fast-math) are automatically applied via .cargo/config.toml
# AVX2/AVX-512 are auto-detected based on CPU capabilities
build-python:
	@echo "Building Python package with ALL optimizations..."
	@echo "  (target-cpu=native + fast-math + auto AVX2/AVX-512 from .cargo/config.toml)"
	maturin develop --release --features python,huge_pages

# Build Python package without optimizations (for compatibility)
build-python-compat:
	@echo "Building Python package (without CPU-specific optimizations)..."
	@RUSTFLAGS="" maturin develop --release --features python

# Build wheel and install (with ALL optimizations)
# Optimizations (target-cpu=native, fast-math) are automatically applied via .cargo/config.toml
# AVX2/AVX-512 are auto-detected based on CPU capabilities
install-python:
	@echo "Building wheel with ALL optimizations..."
	@echo "  (target-cpu=native + fast-math + auto AVX2/AVX-512 from .cargo/config.toml)"
	maturin build --release --features python,huge_pages
	@echo "Installing wheel..."
	pip install --force-reinstall target/wheels/rabitq_rs-*.whl

# Build wheel and install (without CPU-specific optimizations, for compatibility)
install-python-compat:
	@echo "Building wheel (without CPU-specific optimizations)..."
	@RUSTFLAGS="" maturin build --release --features python
	@echo "Installing wheel..."
	pip install --force-reinstall target/wheels/rabitq_rs-*.whl

# Clean Python build artifacts
clean-python:
	@echo "Cleaning Python build artifacts..."
	rm -rf target/wheels/
	rm -rf python/rabitq_rs.egg-info/
	pip uninstall -y rabitq-rs || true

# Run Python integration tests
test-python: build-python
	@echo "Running Python integration tests..."
	python test_python_bindings.py

# Run Rust tests with Python feature
test-rust:
	@echo "Running Rust tests..."
	cargo test --features python

# Run all tests
test: test-rust test-python
	@echo "All tests passed!"

# Run ann-benchmarks locally
bench-local: build-python
	@echo "Running ann-benchmarks locally..."
	@if [ ! -d "../ann-benchmarks" ]; then \
		echo "Cloning ann-benchmarks..."; \
		cd .. && git clone https://github.com/erikbern/ann-benchmarks.git; \
	fi
	@echo "Copying algorithm files..."
	cp -r ann_benchmarks ../ann-benchmarks/algorithms/rabitq_mstg
	@echo "Running benchmark..."
	cd ../ann-benchmarks && python run.py --algorithm rabitq-mstg --dataset fashion-mnist-784-euclidean --runs 1

# Setup Docker benchmark
bench-docker: build-python
	@echo "Setting up Docker benchmark..."
	@if [ ! -d "../ann-benchmarks" ]; then \
		echo "Cloning ann-benchmarks..."; \
		cd .. && git clone https://github.com/erikbern/ann-benchmarks.git; \
	fi
	@echo "Copying algorithm files..."
	cp -r ann_benchmarks ../ann-benchmarks/algorithms/rabitq_mstg
	cp ann_benchmarks/Dockerfile ../ann-benchmarks/install/rabitq_mstg.Dockerfile
	@echo "Building Docker image..."
	cd ../ann-benchmarks && python install.py --algorithm rabitq-mstg
	@echo "Run: cd ../ann-benchmarks && python run.py --algorithm rabitq-mstg --dataset DATASET"

# Run linter
lint:
	@echo "Running clippy..."
	cargo clippy --features python -- -D warnings

# Format code
format:
	@echo "Formatting Rust code..."
	cargo fmt

# Quick smoke test
smoke: build-python
	@echo "Running smoke test..."
	@python3 -c "import numpy as np; from rabitq_rs import MstgIndex; \
		data = np.random.randn(100, 32).astype(np.float32); \
		idx = MstgIndex(32, metric='euclidean'); \
		idx.fit(data); \
		q = np.random.randn(32).astype(np.float32); \
		idx.set_query_arguments(ef_search=50); \
		r = idx.query(q, 10); \
		print(f'✓ Smoke test passed! Found {len(r)} neighbors')"
