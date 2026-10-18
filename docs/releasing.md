# Distribution verification

Releases are cut from `v*` tags. The release workflow (`.github/workflows/verify.yml`)
verifies the source and the built distributions on every supported platform, then
publishes those exact files.

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

Push the tag (`git tag v0.4.0 && git push origin v0.4.0`). When every required job
and the release-assurance aggregate pass, the workflow:

- requires the tag to match the verified distribution version;
- publishes to [PyPI](https://pypi.org/project/qavf/) through trusted publishing
  (OpenID Connect, no stored token; the `pypi` environment accepts `v*` tags only);
- creates the GitHub release with the distributions and their `SHA256SUMS`;
- pushes the `linux/amd64` and `linux/arm64` image `ghcr.io/ahmedaredah/qavf`
  (version, `major.minor` and `latest` tags), built from the verified wheel.

The conda-forge feedstock and the Homebrew tap (`AhmedAredah/homebrew-tap`) follow
each PyPI release; conda-forge's bot opens the update pull request automatically.

Performance measurements are separate from correctness gates. See
[performance benchmarks](performance_benchmarks.md) for
`scripts/benchmark_performance.py`; microbenchmarks do not establish a universal
throughput or hard RSS guarantee.
