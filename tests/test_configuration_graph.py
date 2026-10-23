"""Configuration graph and CLI boundaries."""

from pathlib import Path

import polars as pl
import pytest

from qavf.cli import main
from qavf.configuration import expand_env, read_config_mapping


def test_environment_expansion_preserves_aliases_without_mutating_input() -> None:
    shared = ["${VALUE:-fallback}"]
    value = [shared] * 10
    for _ in range(8):
        value = [value] * 10
    expanded = expand_env(value, {"VALUE": "expanded"})
    current = expanded
    for _ in range(9):
        assert all(child is current[0] for child in current)
        current = current[0]
    assert current == ["expanded"]
    assert shared == ["${VALUE:-fallback}"]


@pytest.mark.parametrize("payload", ["project: &loop [*loop]\n", "project: &loop {x: *loop}\n"])
def test_cyclic_yaml_is_a_configuration_error(tmp_path: Path, payload: str) -> None:
    config = tmp_path / "cyclic.yaml"
    config.write_text(payload)
    with pytest.raises(ValueError, match="cyclic"):
        read_config_mapping(config)
    assert main(["run", "-c", str(config)]) == 2


def test_reserved_compare_option_has_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "table.parquet"
    pl.DataFrame({"id": [1], "x": [1]}).write_parquet(source)
    config = tmp_path / "compare.yaml"
    config.write_text("entry_points: false\nkeys: [id]\nlevels: [records]\n")
    assert main(["compare-datasets", str(source), str(source), "-c", str(config)]) == 2
    assert "--no-plugins" in capsys.readouterr().err


def test_compare_show_zero_keeps_failure_without_retaining_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import qavf.cli as cli

    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    pl.DataFrame({"id": [1, 2], "x": [1, 2]}).write_parquet(left)
    pl.DataFrame({"id": [1, 2], "x": [10, 20]}).write_parquet(right)
    compare = cli.compare_outputs

    def inspected(*args: object, **kwargs: object):
        assert kwargs["max_differences"] == 0
        result = compare(*args, **kwargs)
        assert result.total_differences == 2
        assert result.differences == []
        return result

    monkeypatch.setattr(cli, "compare_outputs", inspected)
    assert main(["compare", str(left), str(right), "--keys", "id", "--show", "0"]) == 1
    assert "value_changed" not in capsys.readouterr().out


def test_compare_rejects_negative_show() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "left", "right", "--keys", "id", "--show", "-1"])
    assert exc.value.code == 2
