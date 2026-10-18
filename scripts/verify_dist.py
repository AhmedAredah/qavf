"""Verify that a freshly built wheel contains exactly the current package files.

Run after ``uv build --clear``. A successful source test run does not establish
that users will install that source; this checks the actual distribution bytes.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_files() -> dict[str, bytes]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in (ROOT / "qavf").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def development_files() -> dict[str, bytes]:
    """The sdist must retain the exact inputs needed to repeat release checks."""
    inputs = (
        "tests", "config", "docs", "scripts", ".github", ".gitattributes", ".gitignore",
        "pyproject.toml", "uv.lock", "README.md", "CHANGELOG.md", "LICENSE",
    )
    result = {}
    for name in inputs:
        base = ROOT / name
        if not base.exists():
            sys.exit(f"Required release input is missing: {name}")
        for path in base.rglob("*") if base.is_dir() else [base]:
            if (
                path.is_file()
                and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)
                and path.suffix not in {".pyc", ".pyo"}
            ):
                result[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return result


def main() -> None:
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sdists = sorted((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit("Build exactly one wheel and sdist with uv build --clear first.")
    expected = package_files()
    with zipfile.ZipFile(wheels[0]) as archive:
        actual = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("qavf/") and not name.endswith("/")
        }
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        )
        sys.exit(f"Wheel mismatch: missing={missing}; extra={extra}; changed={changed}")
    with tarfile.open(sdists[0]) as archive:
        members = {
            member.name.split("/", 1)[1]: member
            for member in archive.getmembers()
            if member.isfile() and "/" in member.name
        }
        sdist_expected = expected | development_files()
        for name, content in sdist_expected.items():
            member = members.get(name)
            handle = archive.extractfile(member) if member else None
            if handle is None or handle.read() != content:
                sys.exit(f"Sdist mismatch: {name}")
        # Hatch adds PKG-INFO; every other member must match a current declared
        # input. A stale test, workflow or helper changes release behavior too.
        unexpected = members.keys() - sdist_expected.keys() - {"PKG-INFO"}
        if unexpected:
            sys.exit(f"Sdist contains stale or unexpected files: {sorted(unexpected)}")
        if any("__pycache__" in name or name.endswith(".pyc") for name in members):
            sys.exit("Sdist contains interpreter cache files")
    print(f"Verified {len(expected)} package files byte-for-byte in wheel and sdist.")
    print(f"Verified {len(sdist_expected) - len(expected)} exact sdist development inputs.")
    for path in [*wheels, *sdists]:
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")


if __name__ == "__main__":
    main()
