"""Thin CLI for strict source, isolated-wheel, and aggregate release checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from release_checks.extensions import install_extensions
from release_checks.gate import aggregate
from release_checks.runner import run_tests
from release_checks.wheel import installed_checks

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extensions = subparsers.add_parser("extensions", help="Install required DuckDB extensions")
    extensions.add_argument("--integration", action="store_true")
    test = subparsers.add_parser("test", help="Run a strict test suite in the active interpreter")
    test.add_argument("--id", required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--optimized", action="store_true")
    test.add_argument("--coverage", action="store_true", help="Record branch coverage JSON")
    test.add_argument("--integration", action="store_true")
    test.add_argument("paths", nargs="*", default=["tests"])
    installed = subparsers.add_parser("installed", help="Install the actual wheel outside source")
    installed.add_argument("--id", required=True)
    installed.add_argument("--output-dir", type=Path, required=True)
    installed.add_argument("--mode", choices=("locked", "floors", "core"), default="locked")
    gate = subparsers.add_parser(
        "aggregate", help="Reject missing, stale, failed or incomplete checks"
    )
    gate.add_argument("--evidence-dir", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--jobs-json", help="GitHub needs object JSON; also accepts QAVF_CI_JOBS env")
    args = parser.parse_args()
    if args.command == "extensions":
        install_extensions(live=args.integration)
        return 0
    if args.command == "test":
        if args.integration:
            os.environ["QAVF_INTEGRATION_MODE"] = "required"
        return run_tests(
            root=ROOT,
            workdir=ROOT,
            python=Path(sys.executable),
            output=args.output,
            check_id=args.id,
            optimized=args.optimized,
            coverage=args.coverage,
            selection="integration" if args.integration else "not integration",
            paths=tuple(args.paths or ["tests"]),
        )
    if args.command == "installed":
        return installed_checks(ROOT, args.output_dir.resolve(), args.id, args.mode)
    raw_jobs = args.jobs_json or os.environ.get("QAVF_CI_JOBS")
    jobs = json.loads(raw_jobs) if raw_jobs else None
    if jobs is not None and not isinstance(jobs, dict):
        parser.error("CI job conclusions must be an object")
    return aggregate(ROOT, args.evidence_dir, args.output, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
