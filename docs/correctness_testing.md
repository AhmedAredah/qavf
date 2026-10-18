# Correctness test design

The assurance suite checks observable QA decisions against independent expected
results. It does not use the implementation under test to calculate its oracle.
Supported representations must preserve verdicts and exact counts; unsupported
precision or semantics must identify an error on the affected check.

## Shared modules

| Module | Responsibility |
| --- | --- |
| `tests/assurance/support/adapters.py` | Create and clean up equivalent eager, lazy, Arrow, Parquet and query inputs; control file/Arrow partition boundaries |
| `tests/assurance/support/oracles.py` | Plain Python missing-value and composite-key counting policies |
| `tests/assurance/support/strategies.py` | Bounded generated input domains with deliberate edge cases |
| `tests/assurance/support/contracts.py` | Invoke the public facade, require completed assessment, and compare stable result signatures |
| `tests/assurance/conftest.py` | Reproducible and exploratory Hypothesis profiles |

Test modules are divided by behavior: missingness, keys, sources, precision,
temporal semantics, regression, failures, audit values, and cache labels. Small
shared helpers use composition rather than a test class inheritance hierarchy.
The support assertion module is registered for pytest rewriting so its checks
remain active under optimized Python.

## Important invariants

- Missing markers, null policies and composite keys yield independently counted
  violations and exemptions, including when every configured check fails.
- Changing row order, column order or physical partition boundaries preserves
  decisions where ordering is not part of the declared contract.
- Float aggregates are checked against exact rational arithmetic on stored values;
  large integers and Decimal values cannot silently narrow to a different decision.
- Timezone-aware values retain physical instants, including daylight-saving folds;
  incompatible or insufficient temporal precision is explicitly rejected.
- Regression comparisons retain exact added/removed/changed counts while bounding
  evidence. Null composite keys, nested changes and large numbers are explicit cases.
- Loader/plugin failures, unsafe retries and incomplete assessments cannot produce
  a successful overall result. Mutation checks detect content changes even when
  file size and timestamps have been restored. Failed report writes are atomic.
- Core-only verification executes with pandas and SciPy actually absent; optional
  statistics must return an actionable unassessed error.

## Profiles

```bash
uv sync --all-extras --locked
uv run --no-sync pytest tests/assurance
uv run --no-sync python -O -m pytest tests/assurance
QAVF_HYPOTHESIS_PROFILE=extended uv run --no-sync pytest tests/assurance
QAVF_HYPOTHESIS_PROFILE=randomized uv run --no-sync pytest tests/assurance
```

The default `ci` profile permits up to 25 generated examples per property and
backend variant, with deterministic generation and no example database. `extended`
raises that budget to 200. `randomized` allows 200 examples with exploratory random
seeds and a local replay database. Hypothesis can use fewer examples for finite
input domains; test case counts are not generated-example counts. Explicit examples
exercise known boundary cases in addition to generation. Preserve any discovered
counterexample as a fixed regression case before modifying production code.

For PowerShell, set `$env:QAVF_HYPOTHESIS_PROFILE = 'extended'` before the command.
Release verification explicitly selects the deterministic `ci` profile and records
that choice. A successful property test is evidence for its defined input domain,
not a proof over every possible input.

See [release assurance](release_assurance.md) for strict release commands and
[integrations](integrations.md) for genuine service tests. Test evidence records the
interpreter and actual dependency versions; branch coverage is reported separately
from statement coverage.

## References

- [Hypothesis introduction](https://hypothesis.readthedocs.io/en/latest/tutorial/introduction.html)
- [Hypothesis settings](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.settings)
- [pytest assertion rewriting](https://docs.pytest.org/en/stable/how-to/assert.html#assertion-rewriting)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
