#!/usr/bin/env bash
# Create a GitHub Release with the GitHub CLI.
# Publishing to crates.io / PyPI is handled by .github/workflows/release.yml
# when the release is published.
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install: https://cli.github.com/" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <version> [gh release create args...]" >&2
  echo "Example: $0 0.1.0" >&2
  echo "Example: $0 0.1.0 --notes 'Bugfix release'" >&2
  exit 1
fi

version="${1#v}"
shift
tag="v${version}"

cargo_version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
py_version="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -n 1)"

if [[ "$cargo_version" != "$version" || "$py_version" != "$version" ]]; then
  echo "Bump versions first: Cargo.toml=$cargo_version pyproject.toml=$py_version expected=$version" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty; commit or stash changes before releasing." >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" ]]; then
  echo "Warning: releasing from branch '$branch' (expected main)." >&2
fi

git fetch --tags origin

if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "Tag already exists: $tag" >&2
  exit 1
fi

git tag -a "$tag" -m "Release $tag"
git push origin "$tag"

if [[ $# -eq 0 ]]; then
  gh release create "$tag" --generate-notes --verify-tag
else
  gh release create "$tag" --verify-tag "$@"
fi

echo "Created release $tag. CI will publish crates.io and PyPI."
