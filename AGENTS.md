# Repository Guidelines

- after implementing a feature use fmt/clippy/test to ensure quality
  ```bash
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  cargo test --all-targets --all-features
  ```
- this repo is hosted on GitHub. you can use `gh` to access it. (ask for help if not logged in)
- this repo uses GitHub CI for continuous integration and release Rust crates.
