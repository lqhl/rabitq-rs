# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core Rust library code for RaBitQ + IVF; MSTG lives under `src/mstg/`.
- `examples/`: runnable Rust examples and benchmarks (e.g., `examples/mstg_quickstart.rs`).
- `python/`: Python package sources; bindings are in `src/python_bindings.rs`.
- `benchmarks/` and `ann_benchmarks_templates/`: benchmark utilities and ann-benchmarks integration.
- `docs/`: reference docs and specs.
- Tests: Rust tests in `src/tests.rs`; Python integration test in `test_python_bindings.py`.

## Environment & Prerequisites
- Rust toolchain: use the version pinned by `rust-toolchain.toml`.
- Python: use `python3` (this environment may not provide a `python` alias).
- Python bindings tooling: install `maturin` before running binding build/test commands.
- Network: commands using `--features python` may download additional crates and can fail in offline/DNS-restricted environments.

## Build, Test, and Development Commands
- `cargo build --release`: build the Rust library with optimizations.
- `cargo test`: run Rust unit tests (no Python feature).
- `cargo test --features python`: run Rust tests including Python-feature code paths.
- `cargo fmt --all --check`: verify formatting in CI/local checks. Use `cargo fmt` to apply formatting.
- `cargo clippy --features python -- -D warnings`: lint Rust code (matches `make lint`).
- `make build-python`: build Python bindings in development mode via maturin.
- `make test`: run Rust + Python tests (wraps `make test-rust` and `make test-python`).
- `python3 test_python_bindings.py`: run Python integration tests directly (requires `make build-python` first).

## Task-to-Command Matrix
- Core Rust change (`src/`, excluding bindings): run `cargo test` and `cargo clippy -- -D warnings` when possible.
- Python bindings change (`src/python_bindings.rs`, `python/`, `test_python_bindings.py`): run `cargo test --features python`, `cargo clippy --features python -- -D warnings`, and `python3 test_python_bindings.py` (or `make test`).
- Formatting-only change: run `cargo fmt` and then `cargo fmt --all --check`.
- Benchmark code/path change: run relevant benchmark command(s) and include before/after metrics in PR notes.

## Coding Style & Naming Conventions
- Rust: standard rustfmt formatting; use idiomatic Rust naming (`snake_case` for functions/modules, `CamelCase` for types).
- Keep SIMD/unsafe-heavy code localized and well-structured; prefer small helpers for readability.
- Python: follow PEP 8; keep public API names aligned with `rabitq_rs` module naming.

## Testing Guidelines
- Biggest improvement to agent performance and sanity: for any reported bug, write a reproducing test first, then fix, then prove the fix via a passing test.
- Run `cargo test` before changes to core Rust code.
- Run `make test` before PRs affecting bindings or benchmarks (requires `maturin` + `python3` + network when new crates are needed).
- Keep tests deterministic (project uses seeded RNGs in examples/tests).
- Prefer regression test names that describe the bug/behavior (for example, `regression_<area>_<bug>`).

## Definition Of Done
- Relevant commands from the matrix above pass locally.
- New/changed behavior is covered by tests (or an explicit reason is documented).
- No unintended API changes in MSTG/bindings paths.
- Formatting/lint checks are clean for touched code paths.
- PR notes include the exact commands run and key outcomes.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and often include a PR/issue number (e.g., "Optimize MSTG Search (#16)").
- PRs should include a concise description, test results (commands run), and note any benchmark changes if relevant.

## Notes & Constraints
- MSTG is experimental; call out behavior changes or API shifts in PR descriptions.
- Python bindings are optional; ensure `--features python` is used when touching binding code.
