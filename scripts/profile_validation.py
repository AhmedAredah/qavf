"""Profile QAVF and measure first versus warm public validation calls."""

import argparse
import cProfile
import importlib.metadata
import io
import json
import pstats
import statistics
import sys
import time
from pathlib import Path
from typing import Any

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--csv", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--backend", choices=["auto", "duckdb", "polars"], default="auto")
args = p.parse_args()
sys.path.insert(0, str(args.source.resolve()))
t = time.perf_counter()
importlib.import_module("qavf")
bare = time.perf_counter() - t
t = time.perf_counter()
from qavf import validate  # noqa: E402 -- this import is the measurement boundary

loaded = time.perf_counter() - t
schema = {
    "columns": {f"c{i}": {"dtype": "int", "ge": 0} for i in range(14)},
    "primary_key": [f"c{i}" for i in range(7)],
}
kwargs = {"backend": args.backend} if args.backend != "auto" else {}
result: dict[str, Any] = {
    "backend": args.backend,
    "bare_import_s": bare,
    "validate_import_s": loaded,
    "call_s": {},
}
configurations: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("serial", {"engine": {"parallel": False, "native_threads": 2}}),
    (
        "selected",
        {
            "engine": {"parallel": False, "native_threads": 2, "run_unconfigured_rules": False},
            "plugins": {"entry_points": False},
            "rules": {f"INP-00{i}": {} for i in range(1, 5)},
        },
    ),
]
for mode, config in configurations:
    samples = []
    for _ in range(6):
        t = time.perf_counter()
        r = validate(str(args.csv), schema=schema, config=config, **kwargs)
        samples.append(time.perf_counter() - t)
        assert any(x.affected_count == 1200 for x in r.results_for("INP-004")), r.results_for(
            "INP-004"
        )
        assert not any(x.status.value == "ERROR" for x in r.results), r.results
    result["call_s"][mode] = {
        "first": samples[0],
        "warm_median": statistics.median(samples[1:]),
        "all": samples,
    }

# Sequential profiling exposes rule internals rather than thread waits.
prof = cProfile.Profile()
prof.enable()
r = validate(
    str(args.csv),
    schema=schema,
    config={"engine": {"parallel": False, "native_threads": 2}},
    **kwargs,
)
prof.disable()
prof.dump_stats(str(args.output.with_suffix(".prof")))
stream = io.StringIO()
pstats.Stats(prof, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(65)
args.output.with_suffix(".txt").write_text(stream.getvalue())
result["modules_after_validation"] = [
    n
    for n in (
        "duckdb",
        "polars",
        "pyarrow",
        "numpy",
        "pandas",
        "scipy",
        "jinja2",
        "yaml",
        "pydantic",
    )
    if n in sys.modules
]
try:
    result["polars_installed_bytes"] = sum(
        Path(f.locate()).stat().st_size
        for f in (importlib.metadata.files("polars") or [])
        if Path(f.locate()).is_file()
    )
except importlib.metadata.PackageNotFoundError:
    result["polars_installed_bytes"] = 0
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
