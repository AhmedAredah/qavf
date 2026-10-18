# Distribution verification

QAVF is unreleased. This repository builds source and wheel distributions; the
verification workflow does not publish them.

Before distributing an update:

1. Set a new version in `qavf/_version.py` and describe the change in `CHANGELOG.md`.
2. Regenerate documentation and the lockfile:
   ```bash
   uv lock
   uv run python scripts/generate_docs.py
   ```
3. Validate and rebuild in a clean output directory:
   ```bash
   uv sync --all-extras --locked
   uv run pytest
   uv run python -O -m pytest
   uv run ruff check .
   uv run mypy qavf scripts
   uv run python scripts/generate_docs.py --check
   uv build --clear
   uv run python scripts/verify_dist.py
   ```
4. Install the wheel into a fresh environment and run the tests outside the source
   tree, as `.github/workflows/verify.yml` does for each `v*` tag (or when started
   manually). This catches missing package data,
   stale code and undeclared dependencies hidden by an editable installation.
   Run both normal and optimized Python; runtime validation must not depend on
   `assert` statements. The workflow covers Python 3.12, 3.13 and 3.14 and installs
   the direct dependency floors from `tests/dependency-floors.txt` on Python 3.12.
5. Distribute the verified wheel or sdist. Preserve the verification output and
   SHA-256 hashes alongside the release.

`verify_dist.py` compares every package file byte-for-byte against both artifacts,
requires the build/verification scripts in the sdist, and rejects bytecode caches.
`uv build --clear` removes stale artifacts before rebuilding. Never reuse an old
wheel because its filename happens to match the source version.

Publishing to a registry is a separate maintainer decision. No automated publishing
configuration or credentials are included.

Performance measurements are separate from correctness gates. See
[performance benchmarks](performance_benchmarks.md) for
`scripts/benchmark_performance.py`; microbenchmarks do not establish a universal
throughput or hard RSS guarantee.
