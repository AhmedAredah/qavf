"""Static backend contracts, discoverable without loading optional dependencies.

This catalog describes built-in capabilities, not installed plugins or whether a
remote service/extension is available. It never imports dataframe engines or
executes plugin discovery. Runtime validation remains authoritative for a given
configuration and input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CORE_RULE_IDS = tuple(f"INP-{number:03d}" for number in (*range(1, 8), 9))
FULL_RULE_IDS = tuple(
    sorted(
        (
            *CORE_RULE_IDS,
            "INP-008",
            "CMP-001",
            "GEO-001",
            "REG-001",
            *(f"REC-{n:03d}" for n in range(1, 6)),
            *(f"STA-{n:03d}" for n in range(1, 3)),
            *(f"SPA-{n:03d}" for n in range(1, 4)),
            *(f"TMP-{n:03d}" for n in range(1, 9)),
        )
    )
)


@dataclass(frozen=True, slots=True)
class Capability:
    """One named facility with its support status and material conditions."""

    name: str
    supported: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Immutable catalog entry; ``to_dict`` returns detached JSON-ready data."""

    backend: str
    required_packages: tuple[str, ...]
    rule_ids: tuple[str, ...]
    sources: tuple[Capability, ...]
    policies: tuple[Capability, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COMMON_SOURCES = (
    Capability(
        "file", True, "Strict CSV, Parquet, JSON/NDJSON; local files and globs, DuckDB URLs."
    ),
    Capability("query", True, "SELECT queries; owned or borrowed DuckDB connections."),
    Capability(
        "arrow", True, "Eager Arrow tables and record batches; managed streams use BatchSource."
    ),
    Capability("pandas", True, "Requires pandas when a pandas frame is supplied."),
    Capability("loader", True, "Explicit Loader callable returning a supported input."),
)
_COMMON_POLICIES = (
    Capability("source_row_budget", True),
    Capability(
        "memory_and_spill_bytes",
        True,
        "Native memory and spill reservations; not a whole-process RSS guarantee.",
    ),
    Capability("evidence_limits", True, "Exact totals remain independent of retained evidence."),
    Capability("report_limits", True),
    Capability("configuration_includes", True),
)

_BACKENDS = {
    "duckdb": BackendCapabilities(
        backend="duckdb",
        required_packages=("duckdb>=1.3.1", "pyarrow>=16"),
        rule_ids=CORE_RULE_IDS,
        sources=(
            *_COMMON_SOURCES,
            Capability(
                "managed_batches",
                True,
                "Owned BatchSource contexts; schema checked for every batch.",
            ),
            Capability(
                "polars",
                True,
                "Optional Polars frames are adapted through DuckDB/Arrow.",
            ),
            Capability("python_source", False, "Use an explicit Loader descriptor."),
            Capability(
                "database_source", True, "Named configured connections; extensions may be required."
            ),
            Capability("excel", False),
            Capability("lossy_readers", False),
            Capability(
                "named_connections_and_secrets",
                True,
                "Configured DuckDB queries and connections; remote extensions may be required.",
            ),
        ),
        policies=(
            *_COMMON_POLICIES,
            Capability("supervised_execution", True, "Serializable inputs; platform limits apply."),
            Capability("rule_timeout", True, "Cooperative DuckDB interrupt per native rule stage."),
            Capability("require", True),
            Capability(
                "hash_inputs",
                True,
                "Local file identities; remote identity remains source-specific.",
            ),
            Capability(
                "source_retries",
                True,
                "Transient failures only; loaders and batch sources must declare replay safety.",
            ),
            Capability("max_spill_files", False),
            Capability("plugins", False),
        ),
        limitations=(
            "Rule parameters support dataset selectors for INP-002 through INP-006 and INP-009.",
            "Remote manifest cardinality is not available.",
            "Statistical, spatial, temporal, regression and comparison rules "
            "need the Polars backend.",
        ),
    ),
    "polars": BackendCapabilities(
        backend="polars",
        required_packages=("duckdb>=1.3.1", "pyarrow>=16", "polars>=1.32"),
        rule_ids=FULL_RULE_IDS,
        sources=(
            *_COMMON_SOURCES,
            Capability("managed_batches", True),
            Capability("polars", True),
            Capability("python_source", True),
            Capability("database_source", True, "May require a DuckDB extension and credentials."),
            Capability("excel", True, "Requires the DuckDB Excel extension."),
            Capability("lossy_readers", True, "Explicit acknowledgement and rejection budgets."),
            Capability("named_connections_and_secrets", True),
        ),
        policies=(
            *_COMMON_POLICIES,
            Capability("supervised_execution", True, "Serializable inputs; platform limits apply."),
            Capability(
                "rule_timeout",
                True,
                "Cooperative cancellation; arbitrary plugin calls may outlive it.",
            ),
            Capability("require", True),
            Capability("hash_inputs", True),
            Capability("source_retries", True),
            Capability("max_spill_files", True),
            Capability("plugins", True),
        ),
        limitations=(
            "Optional statistical methods and coordinate transforms "
            "require the stats/spatial extras.",
            "Plugin capabilities are discovered only by a requested full-backend operation.",
        ),
    ),
}


def backend_capabilities(backend: str = "duckdb") -> BackendCapabilities:
    """Return the built-in contract for ``duckdb`` or ``polars`` without imports.

    Installation status is deliberately separate from feature support. The
    ``auto`` selector is not a backend; inspect both records to compare choices.
    """
    try:
        return _BACKENDS[backend]
    except (KeyError, TypeError):
        raise ValueError("backend must be 'duckdb' or 'polars'") from None


__all__ = ["BackendCapabilities", "Capability", "backend_capabilities"]
