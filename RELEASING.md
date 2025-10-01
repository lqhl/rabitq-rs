# Release Process

This document describes how to release a new version of `rabitq-rs` to crates.io using automated GitHub Actions.

## Prerequisites

### One-Time Setup

1. **Generate a crates.io API token**
   - Visit <https://crates.io/settings/tokens>
   - Click "New Token"
   - Name: `rabitq-rs-github-actions`
   - Scopes: Select "publish-update" (allows publishing new versions)
   - Click "Create"
   - **Copy the token immediately** (it won't be shown again)

2. **Add token to GitHub Secrets**
   - Go to your repository on GitHub
   - Navigate to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `CARGO_REGISTRY_TOKEN`
   - Value: Paste the crates.io token from step 1
   - Click "Add secret"

## Release Steps

### 1. Prepare the Release

```bash
# Ensure your main branch is up to date
git checkout main
git pull origin main

# Run full validation locally
cargo fmt
cargo clippy --all-targets --all-features
cargo test
cargo build --release
```

### 2. Update Version

Edit `Cargo.toml` and bump the version according to [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.2 → 0.1.3): Bug fixes, no breaking changes
- **Minor** (0.1.3 → 0.2.0): New features, backwards compatible
- **Major** (0.2.0 → 1.0.0): Breaking changes

```toml
[package]
version = "0.1.3"  # Update this line
```

### 3. Commit and Tag

```bash
# Commit the version bump
git add Cargo.toml
git commit -m "Bump version to 0.1.3"

# Create a git tag (must start with 'v')
git tag v0.1.3

# Push both the commit and the tag
git push origin main
git push origin v0.1.3
```

### 4. Monitor the Release

1. Go to: <https://github.com/lqhl/rabitq-rs/actions>
2. Watch the "Release" workflow run
3. The workflow will:
   - ✅ Validate the tag matches `Cargo.toml` version
   - ✅ Run `cargo fmt --check`
   - ✅ Run `cargo clippy`
   - ✅ Run `cargo test`
   - ✅ Validate packaging with `cargo package`
   - 📦 Publish to crates.io
   - 🎉 Create a GitHub Release with changelog

### 5. Verify the Release

- Check crates.io: <https://crates.io/crates/rabitq-rs>
- Check GitHub Releases: <https://github.com/lqhl/rabitq-rs/releases>
- Test installation: `cargo add rabitq-rs@0.1.3`

## What Happens Automatically

When you push a tag starting with `v`:

1. **Version Validation**: Ensures `vX.Y.Z` matches `version = "X.Y.Z"` in Cargo.toml
2. **Full Test Suite**: Runs all checks (fmt, clippy, tests, build)
3. **Publish to crates.io**: Automatically runs `cargo publish`
4. **Create GitHub Release**: Generates release notes from git commits since last tag

## Troubleshooting

### Release workflow failed

- Check the GitHub Actions logs for details
- Common issues:
  - Tag version doesn't match Cargo.toml version
  - Tests failing
  - Clippy warnings
  - Missing `CARGO_REGISTRY_TOKEN` secret

### Version already published

If you need to re-release:

1. You **cannot** overwrite a published version on crates.io
2. Delete the git tag: `git tag -d v0.1.3 && git push origin :refs/tags/v0.1.3`
3. Bump to a new version (e.g., v0.1.4)
4. Repeat the release process

### Rollback a release

Releases cannot be deleted from crates.io, but you can:

- Yank the version: `cargo yank --vers 0.1.3`
- This prevents new projects from using it, but existing users can still access it
- Publish a fixed version immediately after

## Pre-release Testing

To test the release process without publishing:

```bash
# Dry-run the package build
cargo package --list

# Check what would be published
cargo publish --dry-run
```

## CI Workflow Files

- `.github/workflows/ci.yml`: Runs on every push/PR (fmt, clippy, test, build)
- `.github/workflows/release.yml`: Runs on version tags (publishes to crates.io)
