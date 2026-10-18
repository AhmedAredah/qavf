"""Small command-line boundary for benchmark planning and isolated worker modes."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .catalog import CASES
from .models import WorkloadOptions
from .planning import PROFILES, variants
from .storage import read_json, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run correctness-checked realistic QAVF benchmarks"
    )
    result.add_argument("--profile", choices=PROFILES, default="smoke")
    result.add_argument("--rows", nargs="+", type=int, help="Row scales per primary input")
    result.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    result.add_argument(
        "--backends", nargs="+", choices=("native", "lazy", "eager", "file", "context")
    )
    result.add_argument("--formats", nargs="+", choices=("parquet", "csv"))
    result.add_argument(
        "--execution-modes",
        nargs="+",
        choices=("in_process", "isolated"),
        default=["in_process"],
        help="Isolation applies to new capacity cases",
    )
    result.add_argument("--repeat", type=int)
    result.add_argument("--warmups", type=int)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--seed", type=int, default=20261016)
    result.add_argument("--partitions", type=int, default=4)
    result.add_argument("--chunk-rows", type=int, default=65536)
    result.add_argument("--evidence-rows", type=int, default=20)
    result.add_argument("--evidence-bytes", type=int, default=65536)
    result.add_argument("--memory-bytes", type=int)
    result.add_argument("--cache-bytes", type=int)
    result.add_argument("--spill-bytes", type=int)
    result.add_argument("--sample-interval", type=float, default=0.01)
    result.add_argument(
        "--timeout", type=float, default=120.0, help="Seconds per worker/fixture job"
    )
    result.add_argument("--report-format", choices=("none", "json", "html", "both"), default="both")
    result.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument(
        "--baseline", type=Path, help="Optional prior source tree, same fixtures/host"
    )
    result.add_argument("--output", type=Path, default=Path("benchmark-results/results.json"))
    result.add_argument(
        "--work-dir", type=Path, help="Parent for isolated temporary benchmark files"
    )
    result.add_argument(
        "--keep-work", action="store_true", help="Retain debugging fixtures/results"
    )
    result.add_argument("--list", action="store_true", help="List cases without generating inputs")
    result.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--prepare", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if args.worker:
        from .worker import main as run_worker

        if not args.result:
            command.error("worker mode requires a result path")
        return run_worker(args.worker, args.result)
    if args.prepare:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from .catalog import prepare_fixture

        if not args.result:
            command.error("fixture mode requires a result path")
        request = read_json(args.prepare)
        manifest = prepare_fixture(request.pop("family"), Path(request.pop("directory")), **request)
        write_json(args.result, {"status": "passed", "manifest": manifest})
        return 0
    if args.list:
        for case in CASES:
            adapters = ", ".join(f"{a}/{b}" for a, b in case.adapters)
            print(f"{case.name}: {adapters}")
        return 0
    if args.output.suffix.lower() != ".json":
        command.error("output must end in .json; a sibling .html report is also written")
    profile = PROFILES[args.profile]
    repetitions = profile["repetitions"] if args.repeat is None else args.repeat
    warmups = profile["warmups"] if args.warmups is None else args.warmups
    if repetitions < 1 or warmups < 0 or args.partitions < 1 or args.chunk_rows < 1:
        command.error("repeat/partitions/chunk-rows must be positive; warmups nonnegative")
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.sample_interval)):
        command.error("timeout and sample-interval must be finite and positive")
    rows = args.rows or [profile["rows"]]
    try:
        options = WorkloadOptions(
            threads=args.threads,
            evidence_rows=args.evidence_rows,
            evidence_bytes=args.evidence_bytes,
            memory_budget_bytes=args.memory_bytes,
            cache_max_bytes=args.cache_bytes,
            spill_max_bytes=args.spill_bytes,
        )
        plan = variants(
            args.cases, rows, options, args.backends, args.formats, args.execution_modes
        )
    except ValueError as error:
        command.error(str(error))
    from .suite import run_suite

    sources = {"candidate": args.candidate.resolve()}
    if args.baseline:
        sources = {"baseline": args.baseline.resolve(), **sources}
    suite = run_suite(
        root=Path(__file__).resolve().parents[2],
        sources=sources,
        plan=plan,
        protocol={
            "profile": args.profile,
            "rows": rows,
            "seed": args.seed,
            "threads": args.threads,
            "repetitions": repetitions,
            "warmups": warmups,
            "sample_interval_seconds": args.sample_interval,
            "report_format": args.report_format,
            "partitions": args.partitions,
            "chunk_rows": args.chunk_rows,
            "execution": "sequential fresh processes; alternating source order",
            "os_page_cache": "uncontrolled",
            "timeout_seconds": args.timeout,
        },
        output=args.output,
        timeout_s=args.timeout,
        work_parent=args.work_dir,
        keep_work=args.keep_work,
    )
    print(f"{suite['status']}: {args.output.resolve()}")
    return 0 if suite["status"] == "passed" else 1
