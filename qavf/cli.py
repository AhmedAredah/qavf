"""Command-line interface.

Exit codes: ``0`` success, ``1`` QA failed (or outputs differ from baseline, or a
dataset comparison failed or was not fully assessed), ``2`` usage, configuration
or resource-budget error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from qavf._version import __version__
from qavf.engine.resources import ResourceLimitError
from qavf.engine.runner import ConfigurationError, QARunner, build_registry, build_source_registry
from qavf.engine.severity import Severity
from qavf.io import atomic_write
from qavf.profiling import profile_input
from qavf.redaction import exception_details
from qavf.regression.compare_outputs import compare_outputs
from qavf.reports.html import write_html
from qavf.scaffold import (
    SCHEMA_FILENAME,
    config_json_schema,
    render_config,
    validate_margin,
    write_json_schema,
)
from qavf.settings import QAVFSettings, check_sql_names, load_settings
from qavf.sources.base import SourceError
from qavf.sources.formats import file_name

logger = logging.getLogger("qavf")

EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE = 0, 1, 2


def _margin(text: str) -> float:
    """argparse type for profiling margins: a finite number >= 0."""
    try:
        return validate_margin(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qavf",
        description="Configurable, plugin-based data QA/QC with reports and regression testing.",
    )
    parser.add_argument("--version", action="version", version=f"qavf {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run QA rules and write reports")
    run.add_argument("-c", "--config", required=True, type=Path, help="YAML or JSON configuration")
    run.add_argument(
        "--html", type=Path, help="HTML report path (default: <output_dir>/qa_report.html)"
    )
    run.add_argument(
        "--json", type=Path, help="JSON report path (default: <output_dir>/qa_report.json)"
    )
    run.add_argument(
        "--fail-on", type=Severity, choices=list(Severity), help="override engine.fail_on"
    )
    run.add_argument("--no-parallel", action="store_true", help="run rules sequentially")

    init = commands.add_parser("init", help="write a complete starting configuration")
    init.add_argument("-o", "--output", type=Path, default=Path("qavf.yaml"))
    init.add_argument(
        "-d",
        "--dataset",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="profile a data file and add it with inferred QA limits (repeatable)",
    )
    init.add_argument("--project", default="QA run")
    init.add_argument(
        "--margin",
        type=_margin,
        default=0.0,
        help="widen inferred bounds by this fraction of the range (>= 0)",
    )
    init.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="largest distinct count turned into an isin list",
    )
    init.add_argument("--no-bounds", action="store_true", help="do not infer ge/le bounds")
    init.add_argument("--no-schema", action="store_true", help=f"do not write {SCHEMA_FILENAME}")
    init.add_argument("--force", action="store_true", help="overwrite existing files")

    schema = commands.add_parser("schema", help="write the configuration JSON Schema")
    schema.add_argument("-o", "--output", type=Path, help="file to write (default: stdout)")
    schema.add_argument("-c", "--config", type=Path, help="configuration (for plugin discovery)")

    profile = commands.add_parser("profile", help="print inferred QA limits for a data file")
    profile.add_argument("path")
    profile.add_argument("--margin", type=_margin, default=0.0, help="fraction of the range (>= 0)")
    profile.add_argument("--max-categories", type=int, default=20)

    rules = commands.add_parser("list-rules", help="list discovered rules")
    rules.add_argument("-c", "--config", type=Path, help="configuration (for plugin discovery)")
    rules.add_argument("--format", choices=["table", "markdown", "json"], default="table")

    compare = commands.add_parser("compare", help="compare an output file with a baseline")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("-k", "--keys", nargs="+", required=True, help="key columns")
    compare.add_argument("--rtol", type=float, default=1e-6)
    compare.add_argument("--atol", type=float, default=1e-9)
    compare.add_argument("--ignore", nargs="*", default=[], help="columns to ignore")
    compare.add_argument("--show", type=int, default=20, help="differences to print")
    compare.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="a baseline column renamed in the candidate (repeatable); keys use NEW",
    )

    datasets = commands.add_parser(
        "compare-datasets", help="compare two datasets at several levels and write reports"
    )
    datasets.add_argument("baseline", help="baseline data file (e.g. the previous version)")
    datasets.add_argument("candidate", help="candidate data file compared with it")
    datasets.add_argument(
        "-c", "--config", type=Path, help="YAML or JSON comparison spec (keys, columns, levels)"
    )
    datasets.add_argument("--html", type=Path, help="HTML report path")
    datasets.add_argument("--json", type=Path, help="JSON report path (the typed comparison)")
    datasets.add_argument("--title", help="HTML page title")
    datasets.add_argument(
        "--no-plugins", action="store_true", help="do not load installed comparison plugins"
    )

    demo = commands.add_parser("demo", help="write a runnable demo project")
    demo.add_argument("directory", type=Path)
    demo.add_argument("--clean", action="store_true", help="no injected defects")
    demo.add_argument("--defects", nargs="+", metavar="NAME", help="inject only these defects")
    demo.add_argument("--force", action="store_true", help="write into a non-empty directory")

    baseline = commands.add_parser(
        "baseline", help="accept an output file as a baseline (data plus a manifest)"
    )
    baseline.add_argument("source")
    baseline.add_argument("destination", type=Path)
    baseline.add_argument(
        "-k", "--keys", nargs="+", required=True, help="key columns identifying each row"
    )
    baseline.add_argument("--reason", help="why this output is accepted (recorded in the manifest)")
    baseline.add_argument(
        "--force", action="store_true", help="replace an existing baseline (its diff is shown)"
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def _settings(path: Path | None) -> QAVFSettings:
    return load_settings(path) if path else QAVFSettings()


def _cmd_run(args: argparse.Namespace) -> int:
    from qavf.reports.presentation import HtmlLimits

    settings = load_settings(args.config)
    if args.no_parallel:
        settings.engine.parallel = False
    if args.fail_on:
        settings.engine.fail_on = args.fail_on

    report = QARunner(settings).run()
    fail_on = settings.engine.fail_on
    output_dir = settings.resolve(settings.report.output_dir)
    written = []
    if settings.report.html or args.html:
        written.append(
            write_html(
                report,
                args.html or output_dir / "qa_report.html",
                settings.report.title,
                fail_on,
                limits=HtmlLimits(
                    max_cells=settings.report.html_max_cells,
                    max_columns=settings.report.html_max_columns,
                    max_output_bytes=settings.report.html_max_bytes,
                ),
            )
        )
    if settings.report.json_ or args.json:
        written.append(report.write_json(args.json or output_dir / "qa_report.json"))

    summary = report.summary
    print(
        f"{summary.total} results: {summary.passed} passed, {summary.failed} failed, "
        f"{summary.errored} could not run, {summary.skipped} skipped"
    )
    for result in report.failures():
        print(f"  [{result.severity.value:<8}] {result.label} {result.name}: {result.message}")
    for path in written:
        print(f"Report: {path}")
    verdict = report.verdict(fail_on)
    if verdict == "incomplete":
        print(
            f"QA INCOMPLETE: {summary.errored} configured check(s) could not be assessed"
            if summary.errored
            else "QA INCOMPLETE: no check assessed any data"
        )
    else:
        print(f"QA {verdict.upper()} (fail on {fail_on.value} or above)")
    return EXIT_QA_FAILED if report.exit_code(fail_on) else EXIT_OK


def _parse_dataset(argument: str) -> tuple[str, str]:
    # A raw URI's query may contain '='; only a leading NAME= is a name separator.
    name, separator, tail = argument.partition("=")
    if separator and name.isidentifier():
        path = tail
    else:
        path = argument
        name = file_name(path).split(".")[0]
    if not name.isidentifier():
        raise ValueError(f"dataset name {name!r} must be a valid identifier; use NAME=PATH")
    if not path.strip():
        raise ValueError("a dataset path must not be blank")
    return name, path


def _cmd_init(args: argparse.Namespace) -> int:
    output: Path = args.output
    if output.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("configuration output must use .yaml, .yml or .json")
    schema_path = output.parent / SCHEMA_FILENAME
    targets = [output] if args.no_schema else [output, schema_path]
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("configuration and JSON Schema outputs must be different paths")
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        raise ValueError(f"refusing to overwrite {', '.join(existing)} (use --force)")

    parsed = [_parse_dataset(argument) for argument in args.dataset]
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError("dataset names must not be repeated")
    check_sql_names(names, "dataset")
    if type(args.max_categories) is not int or args.max_categories < 0:
        raise ValueError("max_categories must be a non-negative integer")
    datasets: dict[str, dict[str, object]] = {}
    for name, path in parsed:
        profile = profile_input(
            path, margin=args.margin, max_categories=args.max_categories, bounds=not args.no_bounds
        )
        relative = (
            path
            if "://" in path
            else Path(os.path.relpath(Path(path).resolve(), output.parent.resolve())).as_posix()
        )
        datasets[name] = {
            "path": relative,
            "schema": profile.schema,
        }
        print(f"Profiled {name}: {profile.rows:,} rows, {profile.columns} columns")

    registry = build_registry(QAVFSettings())
    text = render_config(
        registry,
        datasets,
        project=args.project,
        schema_reference=None if args.no_schema else SCHEMA_FILENAME,
    )
    settings = QAVFSettings.model_validate(yaml.safe_load(text))
    if output.suffix.lower() == ".json":
        payload = settings.model_dump(mode="json", by_alias=True)
        if not args.no_schema:
            payload["$schema"] = SCHEMA_FILENAME
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    artifacts = [(output, text)]
    if not args.no_schema:
        schema = config_json_schema(registry, build_source_registry(QAVFSettings()))
        artifacts.append((schema_path, write_json_schema(schema)))
    # Generate and validate every payload before publishing any output. Atomic
    # writes protect individual files; destination planning prevents self-overwrite.
    for destination, content in artifacts:
        atomic_write(destination, content)
    print(f"Configuration written: {output} ({len(registry)} rules)")
    if not args.no_schema:
        print(f"JSON Schema written: {schema_path}")
    return EXIT_OK


def _cmd_schema(args: argparse.Namespace) -> int:
    settings = _settings(args.config)
    schema = config_json_schema(build_registry(settings), build_source_registry(settings))
    text = write_json_schema(schema)
    if args.output:
        atomic_write(args.output, text)
        print(f"JSON Schema written: {args.output}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def _cmd_profile(args: argparse.Namespace) -> int:
    profile = profile_input(args.path, margin=args.margin, max_categories=args.max_categories)
    sys.stdout.write(
        yaml.safe_dump({"schema": profile.schema}, sort_keys=False, allow_unicode=True)
    )
    return EXIT_OK


def _cmd_list_rules(args: argparse.Namespace) -> int:
    registry = build_registry(_settings(args.config))
    rules = list(registry)
    if args.format == "json":
        payload = [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "category": r.category,
                "default_severity": r.default_severity.value,
                "description": r.description,
                "module": r.__module__,
            }
            for r in rules
        ]
        print(json.dumps(payload, indent=2))
    elif args.format == "markdown":
        print("| ID | Category | Default severity | Name | Description |")
        print("|---|---|---|---|---|")
        for r in rules:
            cells = [r.rule_id, r.category, r.default_severity.value, r.name, r.description]
            print("| " + " | ".join(cells) + " |")
    else:
        print(f"{'ID':<12} {'CATEGORY':<12} {'SEVERITY':<9} NAME")
        for r in rules:
            print(f"{r.rule_id:<12} {r.category:<12} {r.default_severity.value:<9} {r.name}")
        print(f"\n{len(rules)} rules.")
    return EXIT_OK


def _column_map(pairs: Sequence[str]) -> dict[str, str]:
    """``OLD=NEW`` arguments as a mapping; a malformed or repeated name is a usage error."""
    mapping: dict[str, str] = {}
    for pair in pairs:
        old, separator, new = pair.partition("=")
        if not separator or not old or not new:
            raise ValueError(f"--map expects OLD=NEW, got {pair!r}")
        if old in mapping:
            raise ValueError(f"--map names baseline column {old!r} more than once")
        mapping[old] = new
    return mapping


def _cmd_compare(args: argparse.Namespace) -> int:
    result = compare_outputs(
        args.baseline,
        args.candidate,
        args.keys,
        rtol=args.rtol,
        atol=args.atol,
        ignore_columns=args.ignore,
        column_map=_column_map(args.map) or None,
    )
    print(result.summary())
    for difference in result.differences[: args.show]:
        where = ", ".join(f"{k}={v}" for k, v in difference.key.items())
        if difference.kind == "value_changed":
            change = f"{difference.baseline!r} -> {difference.candidate!r}"
            print(f"  [{where}] {difference.column}: {change}")
        else:
            print(f"  {difference.kind}: {where or difference.column}")
    return EXIT_OK if result.equal else EXIT_QA_FAILED


def _comparison_spec(path: Path | None) -> dict[str, Any]:
    """The comparison options of a YAML or JSON spec file (empty without one)."""
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a mapping of comparison options")
    if clash := sorted({"baseline", "candidate"} & set(loaded)):
        raise ValueError(
            f"{path} sets {', '.join(clash)}; the datasets are the command's arguments"
        )
    return loaded


def _cmd_compare_datasets(args: argparse.Namespace) -> int:
    from qavf.compare.api import compare_datasets

    options = _comparison_spec(args.config)
    for argument in (args.baseline, args.candidate):
        local = "://" not in argument and not any(c in argument for c in "*?[")
        if local and not Path(argument).exists():
            raise FileNotFoundError(f"no such data file: {argument}")
    result = compare_datasets(
        args.baseline, args.candidate, entry_points=not args.no_plugins, **options
    )
    written = []
    if args.html:
        written.append(result.write_html(args.html, args.title))
    if args.json:
        written.append(result.write_json(args.json))
    print(result.summary())
    for level in result.levels:
        for check in level.checks:
            if not check.passed:
                state = "NOT ASSESSED" if not check.assessed else "FAILED"
                print(f"  [{state}] {level.name}: {check.check}: {check.message}")
    for path in written:
        print(f"Report: {path}")
    print(f"COMPARISON {result.verdict.upper()}")
    return EXIT_OK if result.exit_code == 0 else EXIT_QA_FAILED


def _cmd_baseline(args: argparse.Namespace) -> int:
    from qavf.regression.baselines import accept_baseline

    target = Path(args.destination)
    if target.exists():
        if not args.force:
            raise ValueError(
                f"{target} already exists; review the change with 'qavf compare' and "
                f"pass --force to replace the accepted baseline"
            )
        # Replacing an accepted baseline is a reviewable change: show what changes.
        print(compare_outputs(target, args.source, args.keys, name=target.name).summary())
    manifest = accept_baseline(
        args.source,
        target,
        args.keys,
        reason=args.reason,
        source=str(args.source),
        replace=args.force,
    )
    print(f"Baseline written: {target} ({manifest.rows:,} rows, sha256 {manifest.sha256[:12]})")
    if not args.reason:
        print("Tip: record why it is correct with --reason (kept in the manifest).")
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace) -> int:
    from qavf.demo import write_demo_project

    target: Path = args.directory
    if target.exists() and any(target.iterdir()) and not args.force:
        raise ValueError(f"{target} is not empty (use --force)")
    config = write_demo_project(target, args.defects, clean=args.clean)
    schema = config_json_schema(
        build_registry(QAVFSettings()), build_source_registry(QAVFSettings())
    )
    (target / SCHEMA_FILENAME).write_text(write_json_schema(schema), encoding="utf-8")
    print(f"Demo project written: {target}")
    print(f"Run it with:  qavf run -c {config}")
    return EXIT_OK


_COMMANDS = {
    "run": _cmd_run,
    "init": _cmd_init,
    "schema": _cmd_schema,
    "profile": _cmd_profile,
    "list-rules": _cmd_list_rules,
    "compare": _cmd_compare,
    "compare-datasets": _cmd_compare_datasets,
    "baseline": _cmd_baseline,
    "demo": _cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``qavf`` console script."""
    args = _parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return _COMMANDS[args.command](args)
    except (
        ConfigurationError,
        ValidationError,
        FileNotFoundError,
        SourceError,
        ResourceLimitError,
        KeyError,
        ValueError,
        OSError,  # unreadable configuration, unwritable report location, ...
    ) as exc:
        message = exception_details(exc)
        logger.debug("Command failed: %s", message)
        print(f"error: {message}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
