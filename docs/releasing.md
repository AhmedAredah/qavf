# Distribution verification

Releases are cut from `v*` tags. The release workflow (`.github/workflows/verify.yml`)
verifies the source and the built distributions on every supported platform and
retains the complete release-assurance evidence. It can also be started manually.
This workflow does not publish packages, create releases, or deploy containers.

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

A `v*` tag starts verification. Inspect the retained `release-assurance` artifact
and require `release_ready: true` before any separately authorized publication.
The workflow does not push a tag, publish to PyPI, create a GitHub release, or
update downstream distribution channels. Preserve the exact verified source
revision, wheel/sdist hashes and complete evidence with any release.

Performance measurements are separate from correctness gates. See
[performance benchmarks](performance_benchmarks.md) for
`scripts/benchmark_performance.py`; microbenchmarks do not establish a universal
throughput or hard RSS guarantee.
