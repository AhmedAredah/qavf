"""Release archive gates reject stale code and stale reproducibility inputs."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import verify_dist


@pytest.fixture
def release_tree(tmp_path: Path, monkeypatch):
    files = {
        "qavf/__init__.py": b"version = 'test'\n",
        "tests/test_contract.py": b"def test_truth(): assert True\n",
        "scripts/verify_release.py": b"# release entry point\n",
        "config/example.yaml": b"datasets: {}\n",
        "docs/release_assurance.md": b"Required checks\n",
        ".github/workflows/verify.yml": b"name: verify\n",
        ".gitattributes": b"* text=auto eol=lf\n",
        ".gitignore": b"dist/\n",
        "pyproject.toml": b"[project]\nname = 'test'\n",
        "uv.lock": b"version = 1\n",
        "README.md": b"Example\n",
        "CHANGELOG.md": b"Changes\n",
        "LICENSE": b"MIT\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "dist").mkdir()
    monkeypatch.setattr(verify_dist, "ROOT", tmp_path)
    return tmp_path, files


def write_archives(root: Path, files: dict[str, bytes], changed: str | None = None) -> None:
    with zipfile.ZipFile(root / "dist/test.whl", "w") as wheel:
        for name, content in files.items():
            if name.startswith("qavf/"):
                wheel.writestr(name, content)
    with tarfile.open(root / "dist/test.tar.gz", "w:gz") as sdist:
        for name, content in files.items():
            payload = b"stale release input\n" if name == changed else content
            info = tarfile.TarInfo("test/" + name)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))


def test_archives_with_exact_release_inputs_pass(release_tree):
    root, files = release_tree
    write_archives(root, files)
    verify_dist.main()


@pytest.mark.parametrize("changed", [
    "qavf/__init__.py", "tests/test_contract.py", "uv.lock",
    ".github/workflows/verify.yml", "scripts/verify_release.py",
])
def test_stale_sdist_inputs_block_release(release_tree, changed):
    root, files = release_tree
    write_archives(root, files, changed)
    with pytest.raises(SystemExit, match="Sdist mismatch"):
        verify_dist.main()


def test_stale_wheel_code_blocks_release(release_tree):
    root, files = release_tree
    write_archives(root, files)
    with zipfile.ZipFile(root / "dist/test.whl", "w") as wheel:
        wheel.writestr("qavf/__init__.py", "stale")
    with pytest.raises(SystemExit, match="Wheel mismatch"):
        verify_dist.main()


@pytest.mark.parametrize("extra", [
    "tests/stale_test.py", "scripts/stale_helper.py", ".github/workflows/stale.yml",
    "qavf/stale.py",
])
def test_extra_sdist_release_input_blocks_release(release_tree, extra):
    root, files = release_tree
    write_archives(root, files | {extra: b"stale input\n"})
    # Keep the wheel valid to exercise the independent source archive gate.
    with zipfile.ZipFile(root / "dist/test.whl", "w") as wheel:
        wheel.writestr("qavf/__init__.py", files["qavf/__init__.py"])
    with pytest.raises(SystemExit, match="Sdist contains stale or unexpected"):
        verify_dist.main()
