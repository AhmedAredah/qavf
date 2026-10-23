"""Verify source and installed-wheel suites on the current release platform."""

from __future__ import annotations

import argparse
from pathlib import Path

from release_checks.platform import verify_platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return verify_platform(Path(__file__).resolve().parents[1], args.output_dir)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
