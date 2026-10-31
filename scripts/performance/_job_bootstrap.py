"""Join the controller's Windows job before launching any benchmark code.

Run with ``python -S``. This bootstrap and the production job helper use only
the standard library, so native imports and descendants happen after ownership.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path


def main() -> int:
    control = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    helper = runpy.run_path(control["job_module"])
    helper["join"](control["job"])
    return subprocess.call(control["argv"])


if __name__ == "__main__":
    raise SystemExit(main())
