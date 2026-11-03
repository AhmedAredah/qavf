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
import signal
import sys
import threading
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qavf._version import __version__
from qavf.engine.severity import Severity
from qavf.optional import OptionalDependencyError, has_polars, require_polars
from qavf.run_control import CancellationToken, RunControl, RunEvent

if TYPE_CHECKING:
    import yaml

    from qavf.configuration import read_config_mapping
    from qavf.engine.compilation import build_registry, build_source_registry
    from qavf.engine.runner import QARunner
    from qavf.io import ProtectedInputs, atomic_write
    from qavf.profiling import profile_input
    from qavf.redaction import exception_details
    from qavf.regression.compare_outputs import compare_outputs
    from qavf.reports.html import write_html
    from qavf.scaffold import config_json_schema, render_config, write_json_schema
    from qavf.settings import QAVFSettings, check_sql_names, load_settings
    from qavf.sources.formats import file_name

# Keep the historical module-level helper names available for callers and
# monkeypatching, without importing a dataframe engine just to parse arguments.
_LAZY_IMPORTS = {
    "yaml": ("yaml", None),
    "ValidationError": ("pydantic", "ValidationError"),
    "ConfigurationError": ("qavf.engine.compilation", "ConfigurationError"),
    "ResourceLimitError": ("qavf.resource_errors", "ResourceLimitError"),
    "SourceError": ("qavf.sources.base", "SourceError"),
    "QARunner": ("qavf.engine.runner", "QARunner"),
    "build_registry": ("qavf.engine.compilation", "build_registry"),
    "build_source_registry": ("qavf.engine.compilation", "build_source_registry"),
    "read_config_mapping": ("qavf.configuration", "read_config_mapping"),
    "ProtectedInputs": ("qavf.io", "ProtectedInputs"),
    "atomic_write": ("qavf.io", "atomic_write"),
    "profile_input": ("qavf.profiling", "profile_input"),
    "exception_details": ("qavf.redaction", "exception_details"),
    "compare_outputs": ("qavf.regression.compare_outputs", "compare_outputs"),
    "write_html": ("qavf.reports.html", "write_html"),
    "config_json_schema": ("qavf.scaffold", "config_json_schema"),
    "render_config": ("qavf.scaffold", "render_config"),
    "validate_margin": ("qavf.scaffold", "validate_margin"),
    "write_json_schema": ("qavf.scaffold", "write_json_schema"),
    "QAVFSettings": ("qavf.settings", "QAVFSettings"),
    "check_sql_names": ("qavf.settings", "check_sql_names"),
    "load_settings": ("qavf.settings", "load_settings"),
    "file_name": ("qavf.sources.formats", "file_name"),
}
SCHEMA_FILENAME = "qavf.schema.json"


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    value = import_module(module)
    if attribute is not None:
        value = getattr(value, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


def _use(*names: str) -> None:
    """Resolve command dependencies without replacing caller overrides."""
    for name in names:
        if name not in globals():
            __getattr__(name)


def _native(args: argparse.Namespace) -> bool:
    backend = getattr(args, "backend", "auto")
    if backend == "duckdb" or (backend == "auto" and not has_polars()):
        return True
    require_polars(f"qavf {args.command} --backend polars")
    return False


logger = logging.getLogger("qavf")

EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE = 0, 1, 2


def _margin(text: str) -> float:
    """argparse type for profiling margins: a finite number >= 0."""
    import math

    try:
        value = float(text)
        if not math.isfinite(value) or value < 0:
            raise ValueError("margin must be a finite number >= 0")
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 0") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be an integer >= 0")
    return value


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
        "--fail-on", type=Severity, choices=list(Severity), help="override run.fail_on"
    )
    run.add_argument("--no-parallel", action="store_true", help="run rules sequentially")
    run.add_argument(
        "--backend",
        choices=["auto", "duckdb", "polars"],
        default="auto",
        help="validation backend (auto: Polars when installed, otherwise DuckDB)",
    )
    run.add_argument(
        "--progress",
        action="store_true",
        help="write one JSON progress event per line to stderr",
    )

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
    schema.add_argument("--backend", choices=["auto", "duckdb", "polars"], default="auto")

    profile = commands.add_parser("profile", help="print inferred QA limits for a data file")
    profile.add_argument("path")
    profile.add_argument("--margin", type=_margin, default=0.0, help="fraction of the range (>= 0)")
    profile.add_argument("--max-categories", type=int, default=20)

    checks = commands.add_parser("list-checks", help="list the checks a configuration can use")
    checks.add_argument("-c", "--config", type=Path, help="configuration (for plugin discovery)")
    checks.add_argument("--format", choices=["table", "markdown", "json"], default="table")
    checks.add_argument("--backend", choices=["auto", "duckdb", "polars"], default="auto")

    migrate = commands.add_parser(
        "migrate", help="convert a QAVF 0.6 configuration (rules by code) to named checks"
    )
    migrate.add_argument("config", type=Path, help="configuration to convert")
    migrate.add_argument("-o", "--output", type=Path, help="write here (default: stdout)")
    migrate.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing; exit 1 if anything would",
    )
    migrate.add_argument(
        "--drop-disabled",
        action="store_true",
        help="leave out rules that were disabled instead of keeping them with enabled: false",
    )

    capabilities = commands.add_parser("capabilities", help="show backend capabilities as JSON")
    capabilities.add_argument(
        "--backend", choices=["duckdb", "polars"], help="one backend (default: both)"
    )

    compare = commands.add_parser("compare", help="compare an output file with a baseline")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("-k", "--keys", nargs="+", required=True, help="key columns")
    compare.add_argument("--rtol", type=float, default=1e-6)
    compare.add_argument("--atol", type=float, default=1e-9)
    compare.add_argument("--ignore", nargs="*", default=[], help="columns to ignore")
    compare.add_argument("--show", type=_nonnegative_int, default=20, help="differences to print")
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
    _use("load_settings", "QAVFSettings")
    return load_settings(path) if path else QAVFSettings()


def _check_report_destinations(html: Path | None, json_path: Path | None) -> None:
    if html is None or json_path is None:
        return
    try:
        same = os.path.normcase(str(html.resolve())) == os.path.normcase(str(json_path.resolve()))
    except RuntimeError as exc:
        # Python 3.12 reports symlink loops as RuntimeError rather than OSError.
        raise ValueError("cannot resolve report output path") from exc
    if same:
        raise ValueError("HTML and JSON reports must use different output paths")


class _ProgressLines:
    """Write each run event as one JSON line on stderr, for programs that watch."""

    def __call__(self, event: RunEvent) -> None:
        print(json.dumps(event.to_dict()), file=sys.stderr, flush=True)


class _InterruptCancels:
    """While running, the first interrupt cancels the run so its report is still written.

    Ctrl-C (SIGINT), SIGTERM and, on Windows, Ctrl-Break (SIGBREAK) request
    cancellation; a second interrupt stops the process at once.
    """

    def __init__(self, token: CancellationToken) -> None:
        self.token = token
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> _InterruptCancels:
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
                number = getattr(signal, name, None)
                if number is not None:
                    self.previous[number] = signal.signal(number, self.handle)
        return self

    def handle(self, number: int, frame: Any) -> None:
        if self.token.cancelled:
            raise KeyboardInterrupt
        print("Cancelling: finishing the report (interrupt again to stop now)", file=sys.stderr)
        self.token.cancel()

    def __exit__(self, *_: object) -> None:
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def _cmd_run(args: argparse.Namespace) -> int:
    from qavf.reports.presentation import HtmlLimits

    _use("load_settings", "ProtectedInputs")
    settings = load_settings(args.config)
    if args.no_parallel:
        settings.engine.parallel = False
    if args.fail_on:
        settings.run.fail_on = args.fail_on

    output_dir = settings.resolve(settings.report.output_dir)
    html_path = args.html
    if html_path is None and settings.report.html:
        html_path = output_dir / "qa_report.html"
    json_path = args.json
    if json_path is None and settings.report.json_:
        json_path = output_dir / "qa_report.json"
    _check_report_destinations(html_path, json_path)
    protected = ProtectedInputs.of_settings(settings, args.config)
    protected.check(html_path, json_path)
    token = CancellationToken()
    on_event = _ProgressLines() if args.progress else None
    in_process = settings.engine.execution_mode == "in_process"
    with _InterruptCancels(token):
        if _native(args):
            from qavf.api import validate

            report = validate(
                config=settings,
                backend="duckdb",
                on_event=on_event if in_process else None,
                cancel=token if in_process else None,
            )
        else:
            _use("QARunner")
            control = RunControl(on_event, token) if in_process else None
            report = QARunner(settings).run(control=control)
    fail_on = settings.run.fail_on
    with protected.publication():
        written = []
        if html_path is not None:
            _use("write_html")
            written.append(
                write_html(
                    report,
                    html_path,
                    settings.report.title,
                    fail_on,
                    limits=HtmlLimits(
                        max_cells=settings.report.html_max_cells,
                        max_columns=settings.report.html_max_columns,
                        max_output_bytes=settings.report.html_max_bytes,
                    ),
                )
            )
        if json_path is not None:
            written.append(report.write_json(json_path))

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
    if report.metadata.cancelled:
        print(f"QA CANCELLED: {summary.errored} configured check(s) were not assessed")
    elif verdict == "incomplete":
        print(
            f"QA INCOMPLETE: {summary.errored} configured check(s) could not be assessed"
            if summary.errored
            else "QA INCOMPLETE: no check assessed any data"
        )
    else:
        print(f"QA {verdict.upper()} (fail on {fail_on.value} or above)")
    return EXIT_QA_FAILED if report.exit_code(fail_on) else EXIT_OK


def _parse_dataset(argument: str) -> tuple[str, str]:
    _use("file_name")
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
    _use(
        "yaml",
        "check_sql_names",
        "profile_input",
        "build_registry",
        "QAVFSettings",
        "render_config",
        "config_json_schema",
        "build_source_registry",
        "write_json_schema",
        "atomic_write",
    )
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
    print(f"Configuration written: {output}")
    if not args.no_schema:
        print(f"JSON Schema written: {schema_path}")
    return EXIT_OK


def _cmd_schema(args: argparse.Namespace) -> int:
    settings = _settings(args.config)
    if _native(args):
        from qavf.cli_core import config_json_schema as core_schema

        schema = core_schema(settings)
        text = json.dumps(schema, indent=2) + "\n"
    else:
        _use("config_json_schema", "build_registry", "build_source_registry", "write_json_schema")
        schema = config_json_schema(build_registry(settings), build_source_registry(settings))
        text = write_json_schema(schema)
    if args.output:
        _use("atomic_write")
        atomic_write(args.output, text)
        print(f"JSON Schema written: {args.output}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def _cmd_profile(args: argparse.Namespace) -> int:
    _use("profile_input", "yaml")
    profile = profile_input(args.path, margin=args.margin, max_categories=args.max_categories)
    sys.stdout.write(
        yaml.safe_dump({"schema": profile.schema}, sort_keys=False, allow_unicode=True)
    )
    return EXIT_OK


def _cmd_list_checks(args: argparse.Namespace) -> int:
    settings = _settings(args.config)
    if _native(args):
        from qavf.cli_core import core_checks

        checks = core_checks(settings)
    else:
        _use("build_registry")
        from qavf.checks.listing import listed_checks

        checks = listed_checks(build_registry(settings))
    if args.format == "json":
        print(json.dumps([check.to_dict() for check in checks], indent=2))
    elif args.format == "markdown":
        print("| Check | Code | Where | Default severity | Name | Description |")
        print("|---|---|---|---|---|---|")
        for check in checks:
            cells = [
                f"`{check.check_name}`",
                check.rule_id,
                check.placement,
                check.default_severity,
                check.name,
                check.description,
            ]
            print("| " + " | ".join(cells) + " |")
    else:
        print(f"{'CHECK':<18} {'CODE':<9} {'WHERE':<9} {'SEVERITY':<9} NAME")
        for check in checks:
            print(
                f"{check.check_name:<18} {check.rule_id:<9} {check.placement:<9} "
                f"{check.default_severity:<9} {check.name}"
            )
        print(
            f"\n{len(checks)} checks. 'schema' checks run from each dataset's schema; "
            "the others are configured in 'checks' lists."
        )
    return EXIT_OK


def _cmd_migrate(args: argparse.Namespace) -> int:
    from qavf.migrate import migrate_text

    source = args.config.resolve()
    migration, text = migrate_text(
        source.read_text(encoding="utf-8"), source, drop_disabled=args.drop_disabled
    )
    if args.check:
        if not migration.changed:
            print(f"{args.config}: already uses named checks")
            return EXIT_OK
        print(f"{args.config}: would change:")
        for note in migration.notes:
            print(f"  - {note}")
        return EXIT_QA_FAILED
    if args.output is None:
        sys.stdout.write(text)
    else:
        _use("atomic_write")
        atomic_write(args.output, text)
        print(f"Configuration written: {args.output}")
    for note in migration.notes:
        print(f"  - {note}", file=sys.stderr)
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
    _use("compare_outputs")
    result = compare_outputs(
        args.baseline,
        args.candidate,
        args.keys,
        rtol=args.rtol,
        atol=args.atol,
        ignore_columns=args.ignore,
        column_map=_column_map(args.map) or None,
        max_differences=args.show,
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
    _use("read_config_mapping")
    loaded, _ = read_config_mapping(path)
    if clash := sorted({"baseline", "candidate"} & set(loaded)):
        raise ValueError(
            f"{path} sets {', '.join(clash)}; the datasets are the command's arguments"
        )
    if "entry_points" in loaded:
        raise ValueError(f"{path} sets entry_points; use the --no-plugins command option instead")
    return loaded


def _cmd_compare_datasets(args: argparse.Namespace) -> int:
    from qavf.compare.api import compare_datasets

    _use("ProtectedInputs")
    _check_report_destinations(args.html, args.json)
    protected = ProtectedInputs([args.config, args.baseline, args.candidate])
    protected.check(args.html, args.json)
    options = _comparison_spec(args.config)
    for argument in (args.baseline, args.candidate):
        local = "://" not in argument and not any(c in argument for c in "*?[")
        if local and not Path(argument).exists():
            raise FileNotFoundError(f"no such data file: {argument}")
    result = compare_datasets(
        args.baseline, args.candidate, entry_points=not args.no_plugins, **options
    )
    with protected.publication():
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

    _use("compare_outputs")
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

    _use(
        "config_json_schema",
        "build_registry",
        "build_source_registry",
        "QAVFSettings",
        "write_json_schema",
    )
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


def _cmd_capabilities(args: argparse.Namespace) -> int:
    from qavf.capabilities import backend_capabilities

    backends = [args.backend] if args.backend else ["duckdb", "polars"]
    print(json.dumps({name: backend_capabilities(name).to_dict() for name in backends}, indent=2))
    return EXIT_OK


_COMMANDS = {
    "run": _cmd_run,
    "capabilities": _cmd_capabilities,
    "init": _cmd_init,
    "schema": _cmd_schema,
    "profile": _cmd_profile,
    "list-checks": _cmd_list_checks,
    "migrate": _cmd_migrate,
    "compare": _cmd_compare,
    "compare-datasets": _cmd_compare_datasets,
    "baseline": _cmd_baseline,
    "demo": _cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``qavf`` console script."""
    args = _parser().parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "capabilities":
        return _COMMANDS[args.command](args)
    from qavf.resource_errors import ResourceLimitError
    from qavf.sources.base import SourceError

    try:
        if args.command in {"init", "profile", "compare", "compare-datasets", "baseline", "demo"}:
            require_polars(f"qavf {args.command}")
        return _COMMANDS[args.command](args)
    except (
        OptionalDependencyError,
        SourceError,
        ResourceLimitError,
        KeyError,
        ValueError,
        OSError,
    ) as exc:
        _use("exception_details")
        message = exception_details(exc)
        logger.debug("Command failed: %s", message)
        print(f"error: {message}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
