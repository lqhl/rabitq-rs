# Agent Task Plan

## Implementation Steps
- [x] Review existing RaBitQ documentation and C++ reference implementation to understand algorithm details.
- [x] Create a new Rust crate for the RaBitQ IVF library with project scaffolding.
- [x] Implement core components: random rotator, RabitQ quantizer, and IVF indexing/search structures.
- [x] Write unit and integration tests covering quantization accuracy and IVF+RaBitQ search functionality.
- [x] Run `cargo fmt` and `cargo test`, then ensure the workspace is clean.
- [x] Study the C++ fastscan pipeline to map the distance estimation and pruning factors required in Rust.
- [x] Extend the Rust IVF searcher with a fastscan-style lower-bound check before reconstructing candidates.
- [x] Add regression tests that compare fastscan search results against the naive reconstruction path for both L2 and IP metrics.
- [x] Re-run `cargo fmt` and `cargo test`, ensuring the tree is clean afterward.
- [x] Mirror the extended RaBitQ (RaBitQ+) data path from the C++ reference by storing packed ex-codes and their scaling factors in the Rust quantizer and IVF searcher.
- [x] Add focused tests that exercise the extended-code pruning path and confirm it activates while preserving exact search parity.
- [x] Rework the IVF search path to consume the quantized distance estimators directly (without reconstruction) so the Rust behavior matches the C++ fastscan/booster pipeline.
- [x] Update the naive verifier and diagnostics to operate on quantized estimates and keep the regression suite aligned with the C++ semantics.

## Current Task Steps
- [x] Verify the Rust IVF + RaBitQ pipeline against the gist dataset as described in `example.sh`.
- [x] Address any issues uncovered during the dataset-backed test, including implementing missing functionality and adding regression tests.
- [x] Refresh the project documentation by archiving the existing `README.md` to `README.origin.md` and authoring a new README that explains usage and testing.
- [x] Run `cargo fmt`, `cargo clippy`, and `cargo test` after code changes, ensuring the repository is clean.
- [x] Extend the `gist` CLI with an in-crate training mode (`--nlist`) and document how to evaluate the dataset without precomputed centroids.

## Active Task Steps
- [x] Analyse the `kmeans` crate API and determine how to integrate it as a replacement for the custom trainer.
- [x] Swap out `src/kmeans.rs` for a wrapper around the third-party crate, ensuring deterministic seeding and compatibility with the existing IVF pipeline and tests.
- [x] Update or expand regression tests if needed so that k-means backed flows remain covered.
- [x] Run `cargo fmt`, `cargo clippy --all-targets --all-features`, and `cargo test` to validate the workspace.
- [x] Execute README option 1 and option 2 after the change and capture recall / QPS for comparison against the C++ baseline.
