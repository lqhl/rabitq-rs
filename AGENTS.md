- If you encounter something surprising or confusing in this project, add it into this file (AGENTS.md).
- If a test fails, investigate and fix the issue. Don't just ignore it.

- After implementing a feature use fmt/clippy/test to ensure quality
- This repo is hosted on GitHub. You can use `gh` to access it. (ask if not logged in)
- This repo uses GitHub CI (defined in `.github/workflows`) for continuous integration and release Rust crates.

- Surprising finding (2026-02-25): IVF search path applies extended-code refinement (`ex_codes_packed`, `f_add_ex`, `f_rescale_ex`) after FastScan lower-bound filtering, but current MSTG `search_posting_list_fastscan` only uses FastScan estimated distances and does not consume posting-list extended-code fields during search. `src/mstg/posting_list.rs` stores these fields, yet `src/mstg/search.rs` does not currently use them.
- Update (2026-02-25): MSTG now includes stage-2 extended-code refinement in `src/mstg/search.rs` using `ex_codes_packed`/`f_add_ex`/`f_rescale_ex` (SIMD packed-dot when ex_bits is supported, scalar unpack fallback otherwise), with bounded rerank size (`max(64, 8*top_k)` per posting list) to control latency.
- Surprising finding (2026-02-25): After aligning MSTG with IVF-style online `distk` pruning (global top-k heap across posting lists), MSTG-mem latency improved versus the first two-stage rerank patch, but MSTG-disk latency regressed significantly (roughly 4-5x slower than MSTG-mem), likely because mmap posting-list decode + sequential heap updates limit parallelism and cache locality.
- Update (2026-02-26): Introducing a two-stage scheduler (serial bootstrap lists to warm distk, then parallel per-posting-list local heaps with merge) recovered most performance while keeping IVF-aligned lower-bound+ex-refine semantics. On cohere_100k_768d / epsilon=0.02, MSTG-mem balanced improved from ~9073us to ~3164us and MSTG-disk balanced from ~40529us to ~12787us at similar recall (~97.8%).
- Update (2026-02-26): Added MSTG `SearchDiagnostics` with `search_with_diagnostics()` to expose selected centroids, scanned vectors, lower-bound skips, refined vectors, and timing breakdown (`posting_access_time_us`, `posting_search_time_us`, approximate decode/build overhead).
