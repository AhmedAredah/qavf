# Correctness and release assurance

A passing development test run is one piece of release evidence. The release gate
requires complete, current results for each supported environment and for the
actual wheel. Missing jobs, unavailable services, unexpected skips, expected
failures, collection errors and unfinished test runs block readiness. This
workflow does not publish a release.

## Modular verification tools

`scripts/verify_platform.py` runs the complete checks for one host;
`scripts/verify_release.py` exposes individual checks. The
`scripts/release_checks` package separates responsibilities:

- `provenance.py`: source/lock identity, interpreter and dependency versions,
  imported package location and atomic JSON records.
- `policy.py`: narrowly defined inapplicable-test exceptions.
- `pytest_plugin.py`: collection and per-test outcome recording.
- `runner.py`: process execution, crash handling and source-change detection.
- `coverage.py`: run-owned branch coverage and subprocess evidence, including isolated workers.
- `wheel.py`: isolated wheel installation and test-fixture staging.
- `extensions.py`: required signed DuckDB extension provisioning through HTTPS.
- `validation.py`: independent checks of completed outcomes and actual runtimes.
- `gate.py`: the complete required matrix and aggregate release decision.
- `platform.py`: host detection, full command sequencing and retained failure evidence.

These development helpers are included in the source distribution; they add no
runtime dependency to the QAVF wheel. The production API remains unchanged.

## Required matrix

| Environment | Required execution |
| --- | --- |
| Linux / Python 3.12, 3.13, 3.14 | Source and installed wheel, normal and optimized Python; full branch coverage on 3.12 source |
| Windows / Python 3.12 | Source and installed wheel, normal and optimized Python |
| macOS / Python 3.12 | Source and installed wheel, normal and optimized Python |
| Linux / Python 3.12 / direct dependency floors | Installed wheel, normal and optimized Python |
| Linux / Python 3.12 / core dependencies only | Installed wheel capability checks, normal and optimized Python |
| Linux / Python 3.12 / live integration dependencies | Required HTTP, PostgreSQL and S3-compatible service tests |

The included [CI workflow](../.github/workflows/ci.yml) runs on pushes, pull
requests and manual dispatch. Every matrix member runs `verify_platform.py`:
full source and isolated installed-wheel suites in normal and optimized Python,
with distribution validation and signed extension preparation. Linux 3.12 also
captures branch/subprocess coverage and checks lint, types, generated docs and
a separately installed core-only wheel. Matrix jobs do not fail fast; completed
evidence and distributions are uploaded even after failure. The workflow has
read-only repository permissions and does not publish packages.

The workflow has not been executed on GitHub Actions as part of this local
review. Actual Windows and macOS runs remain required. Dependency floors and
live integrations use the separate procedures below. A green routine CI run
does not replace the complete aggregate gate.

Each record identifies the source digest, lockfile digest, Git revision (when
available), CI run/attempt, dependency versions, OS, Python version, optimization
mode and imported QAVF location. Aggregation rejects stale evidence, duplicates,
mislabelled runtimes, wrong test selections and summary counts inconsistent with
completed outcomes. Source text checkout line endings are normalized when hashed;
`.gitattributes` enforces LF for ordinary text checkout files.

## Reproduce locally

Use the interpreter for the desired matrix member; the examples below select
Linux / Python 3.12. Keep evidence outside tracked release inputs.

```bash
uv sync --all-extras --locked --python 3.12
uv run --no-sync python scripts/verify_release.py extensions
uv run --no-sync python scripts/verify_release.py test --coverage --id linux-3.12-source --output evidence/linux-3.12-source.json
uv run --no-sync python scripts/verify_release.py test --optimized --id linux-3.12-source-optimized --output evidence/linux-3.12-source-optimized.json
uv run --no-sync ruff check .
uv run --no-sync mypy qavf scripts
uv run --no-sync python scripts/generate_docs.py --check
uv build --clear
uv run --no-sync python scripts/verify_dist.py
uv run --no-sync python scripts/verify_release.py installed --id linux-3.12-wheel --output-dir evidence
uv run --no-sync python scripts/verify_release.py installed --mode floors --id floors --output-dir evidence
uv run --no-sync python scripts/verify_release.py installed --mode core --id core --output-dir evidence
```

The installed command creates a fresh cross-platform virtual environment and a
separate test directory under the system temporary directory. It copies test and
documentation fixtures, never the `qavf` source package, and verifies the imported
package is inside the wheel environment. Locked runs export the committed lock
with `uv export --locked` and install those exact resolved versions with the wheel.
The committed lock currently selects DuckDB 1.5.6. The updated-package audit ran
locally with DuckDB 2.0.0; that result is separate evidence and does not establish
that the locked 1.5.6 environment passes. The CI workflow uses `uv sync --locked`,
and each strict execution record reports its actual dependency versions.
The floor run intentionally uses the separate direct-version constraints in
`tests/dependency-floors.txt`, not the current dependency lock. Its transitive
versions are recorded; it does not claim they are minimum or locked versions.
On older DuckDB versions, the extension helper bootstraps the signed `httpfs`
binary through Python HTTPS and a temporary local `INSTALL` when DuckDB's own
HTTPS transport would recursively require an HTTP download. TLS, extension
signatures and version/platform checks remain enabled. Download failure still
blocks the gate.

Core-only runs export only locked core dependencies and run explicit checks even
under `python -O`. They require pandas, SciPy and pyproj to be absent, exercise
ordinary input verdicts and exact temporal readback, and verify that unavailable
statistics and unsupported coordinate systems fail explicitly. The floor recipe
includes every core and optional direct dependency, including the spatial extra.

For required integration tests, provision the disposable services and environment
variables described in `docs/integrations.md`, then run:

```bash
uv sync --all-extras --group integration --locked
uv run --no-sync python scripts/verify_release.py extensions --integration
uv run --no-sync python scripts/verify_release.py test --integration --id integrations --output evidence/integrations.json tests/integration
```

The ordinary strict suite deselects `integration` tests; the dedicated integration
phase requires them. A developer's optional integration skips never satisfy release
readiness. Extension download or service failures remain failures, not successful
coverage. External pytest flags, plugin injection and Python import paths are
removed from verification child environments; the required Hypothesis and release
plugins are explicitly loaded. The strict runner forces the reproducible `ci`
Hypothesis profile. `--coverage` emits branch coverage as a sibling
`.coverage.json` file without repeating the suite. The Linux 3.12 source gate
requires this evidence and executed lines from the actual isolated worker.
Coverage uses the locked coverage.py subprocess patch and a small explicit
startup hook for workers launched with `python -S`, after OS limits are installed.
Successful ordinary and isolated children flush coverage; forcibly killed workers
cannot flush their final observations. Coverage is measured across the complete
source suite, not only a focused policy subset. The gate requires real capture;
it does not equate any percentage with correctness or impose an arbitrary threshold.

An optimized test run covers the parent API and its subprocess test driver under
`python -O`. Isolated QAVF workers intentionally run with optimization level zero,
including when the caller exports `PYTHONOPTIMIZE=1` or `2`, so worker assertions
remain enabled. Tests read both actual interpreter flags instead of inferring
them from the parent command. This is the same worker policy in ordinary and
optimized release jobs.

After collecting **all** required environments for the same source revision:

```bash
uv run --no-project python scripts/verify_release.py aggregate --evidence-dir evidence --output release-assurance.json
```

A partial local run deliberately produces `release_ready: false`. When automating
the complete release gate, also supply independent upstream job conclusions via
`QAVF_CI_JOBS` or `--jobs-json`; the included routine CI does not invoke this gate.
Never describe unexecuted
Windows, macOS, remote services, or Python versions as passing based on another
platform's result. Evidence is execution provenance, not cryptographic attestation
against a malicious actor who can modify the workflow and records together.

## Accepted inapplicable branches

No arbitrary allow-skip flag is provided. Four existing test families have narrow
exceptions, checked against the recorded runtime capabilities and exact reason:

1. The deliberate missing-SciPy branch when SciPy is installed. The core-only gate
   independently exercises the missing-dependency behavior.
2. Literal filesystem names forbidden or normalized by Windows (`?`, `*`, control
   characters, newlines, trailing spaces), only on Windows.
3. NumPy extended scalar recursion cases when that dtype becomes an ordinary
   Python scalar on the recorded platform.
4. Platform enforcement internals that do not exist on a host: Linux procfs
   group probes on Windows/macOS, POSIX symlink/FIFO/rlimit cases on Windows, and
   address-space allocation on macOS. Each host runs its own real isolation tests,
   and public tests require explicit rejection of each control it cannot enforce.

An unexpected skip, any xfail/xpass, an empty selection or a skipped collection
fails strict verification. Adding an exception requires changing and testing the
policy explicitly.

## References

- [pytest: exit codes](https://docs.pytest.org/en/stable/reference/exit-codes.html)
  and [skipping/expected failures](https://docs.pytest.org/en/stable/how-to/skipping.html).
- [uv: exporting lockfiles](https://docs.astral.sh/uv/concepts/projects/export/)
  and [locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
- [GitHub Actions: job dependencies](https://docs.github.com/en/actions/using-jobs/using-jobs-in-a-workflow)
  and [required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
- [DuckDB: installing extensions](https://duckdb.org/docs/stable/extensions/installing_extensions.html).
- [coverage.py: subprocess measurement](https://coverage.readthedocs.io/en/latest/subprocess.html).


## Full native platform verification

Use Python 3.12 on Windows and macOS. From the repository root,
run these commands in PowerShell or Terminal:

```sh
uv sync --locked --all-extras --group dev --python 3.12
uv run --no-sync python scripts/verify_platform.py --output-dir release-evidence/attempt-1
```

Keep the entire `release-evidence/attempt-1` directory as that host's evidence. The command builds the
package and runs full source and isolated installed-wheel suites in normal and
optimized Python. Linux supports Python 3.12, 3.13 and 3.14; select the matching
interpreter during sync. Linux 3.12 also records branch and subprocess coverage.
Dependencies and signed DuckDB extensions require network access. Failures and
missing evidence remain blocking. Use a fresh attempt directory for retries;
run only one verification process per checkout because it rebuilds `dist`.
A host pass does not qualify the complete release: dependency floors and live
integrations remain separate gates. Native Windows/macOS results must come
from actual runs on those operating systems.
