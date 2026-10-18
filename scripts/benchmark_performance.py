"""Run modular, realistic QA/QC benchmarks; use --help for workload and scale options."""

from __future__ import annotations

if __package__:
    from .performance.cli import main
else:
    from performance.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
