"""Install and exercise the actual distribution in an isolated temporary directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .provenance import file_digest, repository_identity, timestamp, write_json
from .runner import clean_environment, run_tests

TEST_INPUTS = (
    "tests",
    "scripts",
    "config",
    "docs",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
)
FLOOR_TOOLS = ("pytest==8.0.0", "statsmodels==0.14.1", "jsonschema==4.21.0", "hypothesis==6.140.0")


def venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str], root: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=clean_environment(), check=True)


def installed_checks(root: Path, output_dir: Path, check_id: str, mode: str) -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to create the isolated wheel environment")
    wheels = sorted((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Build exactly one wheel with uv build --clear before verification")
    wheel = wheels[0].resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_path = output_dir / f"{check_id}-setup.json"
    record = {
        "schema_version": 1,
        "id": f"{check_id}-setup",
        "kind": "installation",
        "status": "running",
        "started": timestamp(),
        "identity": repository_identity(root),
        "mode": mode,
        "wheel": wheel.name,
        "wheel_sha256": file_digest(wheel),
    }
    write_json(setup_path, record)
    try:
        with tempfile.TemporaryDirectory(prefix="qavf-wheel-") as temporary:
            directory = Path(temporary)
            if directory.is_relative_to(root):
                raise RuntimeError("Temporary installation directory must be outside source tree")
            envdir = directory / "environment"
            python = venv_python(envdir)
            workdir = directory / "verification"
            workdir.mkdir()
            run([uv, "venv", str(envdir), "--python", sys.executable], root)
            requirements = directory / "locked.txt"
            export = [
                uv,
                "export",
                "--locked",
                "--no-emit-project",
                "--no-hashes",
                "--format",
                "requirements.txt",
                "--output-file",
                str(requirements),
            ]
            export += ["--no-dev"] if mode == "core" else ["--all-extras"]
            if mode != "floors":
                run(export, root)
                run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "-r",
                        str(requirements),
                        str(wheel),
                    ],
                    root,
                )
            else:
                run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "-c",
                        str(root / "tests/dependency-floors.txt"),
                        str(wheel) + "[stats,pandas]",
                        *FLOOR_TOOLS,
                    ],
                    root,
                )
            run([uv, "pip", "check", "--python", str(python)], root)
            for name in TEST_INPUTS:
                source = root / name
                target = workdir / name
                if source.is_dir():
                    shutil.copytree(
                        source,
                        target,
                        ignore=shutil.ignore_patterns(
                            "__pycache__", ".pytest_cache", ".hypothesis", "*.pyc"
                        ),
                    )
                else:
                    shutil.copy2(source, target)
            probe = subprocess.run(
                [str(python), "-I", "-c", "import qavf; print(qavf.__file__)"],
                cwd=workdir,
                env=clean_environment(),
                capture_output=True,
                text=True,
                check=True,
            )
            import_path = Path(probe.stdout.strip()).resolve()
            if not import_path.is_relative_to(envdir.resolve()) or import_path.is_relative_to(root):
                raise RuntimeError(f"qavf was not imported from the isolated wheel: {import_path}")
            record["qavf_import_path"] = str(import_path)
            failures = 0
            if mode != "core":
                try:
                    run(
                        [str(python), str(workdir / "scripts/verify_release.py"), "extensions"],
                        workdir,
                    )
                except subprocess.CalledProcessError as exc:
                    # Keep collecting strict test evidence, but never erase setup failure.
                    record["extension_preparation"] = {
                        "status": "failed",
                        "exit_code": exc.returncode,
                        "error": str(exc),
                    }
                    failures += 1
            if mode == "core":
                run([str(python), "-I", str(workdir / "scripts/verify_core.py")], workdir)
                run([str(python), "-I", "-O", str(workdir / "scripts/verify_core.py")], workdir)
                probe_env = subprocess.run(
                    [
                        str(python),
                        "-c",
                        "import json; from scripts.release_checks.provenance "
                        "import environment; print(json.dumps(environment()))",
                    ],
                    cwd=workdir,
                    env=clean_environment(),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                record["environment"] = json.loads(probe_env.stdout)
                record["verification_modes"] = [0, 1]
            else:
                for optimized in (False, True):
                    phase = check_id + ("-optimized" if optimized else "")
                    failures += run_tests(
                        root=root,
                        workdir=workdir,
                        python=python,
                        output=output_dir / f"{phase}.json",
                        check_id=phase,
                        optimized=optimized,
                        metadata={"installation": mode, "wheel_sha256": file_digest(wheel)},
                    )
            record["status"] = "failed" if failures else "passed"
            if record["identity"] != repository_identity(root):
                record.update(status="failed", error="Source changed during installed verification")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["finished"] = timestamp()
        write_json(setup_path, record)
    return 0 if record["status"] == "passed" else 1
