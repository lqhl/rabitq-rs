# Repository Guidelines

## Project Structure & Module Organization
- Core Rust code sits in `src/`, with `lib.rs` re-exporting modules such as `ivf`, `quantizer`, `rotation`, `math`, and `kmeans`.
- The CLI entry point `src/bin/gist.rs` reproduces the IVF + RaBitQ benchmark; supporting scripts live under `python/`.
- Shared fixtures and regression tests are defined in `src/tests.rs`; large datasets are expected in `data/` and are ignored by git.
- `docs/` and `sample/` contain reference material and lightweight examples you can cite in discussions or tests.

## Build, Test, and Development Commands
- `cargo fmt` formats Rust sources; run it before every commit.
- `cargo clippy --all-targets --all-features` must pass cleanly; treat warnings as action items.
- `cargo test` executes the seeded unit and integration suite in `src/tests.rs`.
- `cargo run --release --bin gist -- --help` lists benchmark options; replace `--help` with dataset flags (see README) to verify end-to-end flows.
- `python python/ivf.py <base.fvecs> <nlist> <out_centroids> <out_assignments> <metric>` mirrors the FAISS tooling when you need pre-clustered inputs.

## Coding Style & Naming Conventions
- Follow rustfmt defaults (4-space indent, trailing commas where helpful) and keep modules focused; split helpers into `io`, `kmeans`, or `quantizer` instead of adding monoliths.
- Prefer `snake_case` for functions and variables, `CamelCase` for types, and uppercase identifiers for constants and metrics.
- Annotate new public APIs with concise rustdoc comments, and gate experimental code behind clearly named feature flags if needed.

## Testing Guidelines
- Add deterministic tests that mirror the patterns in `src/tests.rs`, seeding RNGs via `StdRng::seed_from_u64` to keep runs repeatable.
- Cover both L2 and inner-product paths when touching IVF or quantizer logic; extend the fastscan parity tests when algorithms change.
- For dataset-backed verification, retain small caps (`--max-base`, `--max-queries`) so CI-friendly smoke runs stay fast.

## Commit & Pull Request Guidelines
- Past commits use short, Title Case subjects (e.g., “Rewrite K-Means Trainer”); keep messages under 72 characters and describe the behavior change.
- Reference related issues or benchmarks in the body, and list the commands you ran (fmt/clippy/test/bench).
- PRs should summarize the rationale, note data requirements, and include representative output snippets from `cargo test` or the `gist` binary when relevant.

## Data & Benchmark Tips
- Never version large datasets; add download instructions or scripts instead.
- Document any local hardware requirements (e.g., AVX2 assumptions) in your PR if they influence performance-sensitive code.
