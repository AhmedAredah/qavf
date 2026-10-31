# Releasing

Releases are cut from `v*` tags. Pushing a tag runs `.github/workflows/verify.yml`:
the test suite on Linux, macOS and Windows with Python 3.12, 3.13 and 3.14, the
installed wheel, the dependency floors from `tests/dependency-floors.txt`, a
core-only installation and the live-service integrations. Only when every job
passes does it publish the verified wheel and sdist to PyPI, create the GitHub
release with SHA-256 hashes, and push the `ghcr.io/ahmedaredah/qavf` image.

Before tagging:

1. Set the version in `qavf/_version.py` and describe the change in `CHANGELOG.md`.
2. Check the release locally:
   ```bash
   uv lock
   uv sync --all-extras --locked
   uv run pytest
   uv run python -O -m pytest
   uv run ruff check . && uv run ruff format --check .
   uv run mypy qavf scripts
   uv run python scripts/generate_docs.py --check
   uv build --clear
   uv run python scripts/verify_dist.py
   ```
3. Push the commit, then the tag: `git tag v1.2.3 && git push origin v1.2.3`.

`verify_dist.py` compares every package file byte for byte against both artifacts
and rejects bytecode caches. The publish step refuses a tag that does not match
the built version.
