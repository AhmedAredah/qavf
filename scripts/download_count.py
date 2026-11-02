"""Write a shields.io endpoint badge with QAVF's total downloads.

Sums PyPI (pypistats, without mirrors), conda-forge, the ghcr.io container image
and GitHub release assets. A source that cannot be read counts as zero and is
reported on stderr, so one unavailable service never hides the others.
Uses only the standard library so the scheduled workflow needs no installation.

    python scripts/download_count.py downloads.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY = "AhmedAredah/qavf"
_TIMEOUT_S = 30


def _get(url: str, *, token: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "qavf-download-badge"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return bytes(response.read())


def _json(url: str, *, token: str | None = None) -> Any:
    return json.loads(_get(url, token=token))


def pypi() -> int:
    """Downloads recorded by pypistats since the package was first tracked."""
    rows = _json("https://pypistats.org/api/packages/qavf/overall?mirrors=false")["data"]
    return sum(int(row["downloads"]) for row in rows)


def conda_forge() -> int:
    files = _json("https://api.anaconda.org/package/conda-forge/qavf").get("files", [])
    return sum(int(item.get("ndownloads", 0)) for item in files)


def github_releases() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    releases = _json(
        f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=100", token=token
    )
    return sum(int(asset["download_count"]) for item in releases for asset in item["assets"])


def container() -> int:
    """The package page's total; GitHub's API does not expose container pulls."""
    owner, name = REPOSITORY.split("/")
    page = _get(f"https://github.com/{owner}/{name}/pkgs/container/{name}").decode("utf-8")
    text = re.sub(r"<[^>]+>", " ", page)
    match = re.search(r"Total downloads\s+([\d,.]+\s*[kKmM]?)", text)
    if match is None:
        raise ValueError("total downloads not found on the package page")
    return _expand(match.group(1))


def _expand(count: str) -> int:
    """Expand GitHub's abbreviated counts such as ``1.2k``."""
    count = count.replace(",", "").strip()
    scale = {"k": 1_000, "m": 1_000_000}.get(count[-1:].lower(), 1)
    return round(float(count.rstrip("kKmM")) * scale)


SOURCES: dict[str, Callable[[], int]] = {
    "pypi": pypi,
    "conda-forge": conda_forge,
    "ghcr": container,
    "github-releases": github_releases,
}


def total() -> int:
    downloads = 0
    for name, read in SOURCES.items():
        try:
            count = read()
        except Exception as exc:  # A failed source counts as zero, visibly.
            print(f"{name}: unavailable ({exc})", file=sys.stderr)
            continue
        print(f"{name}: {count}", file=sys.stderr)
        downloads += count
    return downloads


def _label(count: int) -> str:
    for size, suffix in ((1_000_000, "M"), (1_000, "k")):
        if count >= size:
            return f"{count / size:.1f}".rstrip("0").rstrip(".") + suffix
    return str(count)


def badge(count: int) -> dict[str, Any]:
    return {"schemaVersion": 1, "label": "downloads", "message": _label(count), "color": "blue"}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    count = total()
    if count == 0:
        print("no source could be read; keeping the previous badge", file=sys.stderr)
        return 1
    Path(argv[0]).write_text(json.dumps(badge(count)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
