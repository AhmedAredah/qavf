"""The native and Polars backends report the same results for the same contract.

Switching ``backend`` must not change a report: the same rules, check identities,
labels, messages, counts, evidence rows (including ``violates``, ``_file`` and
``_file_row``) and summaries. Only timings may differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qavf import Dataset, Query, validate
from qavf.input_rules import INPUT_RULES

# Timings legitimately differ between engines.
_VOLATILE = {"execution_time_s", "load_time_s"}
# Deliberate scope: the native backend implements the input rules only. The full
# backend also reports its other rule families (GEO, STA, TMP, ...), SKIPPED when
# unconfigured; those results are excluded from the comparison, and the native
# backend must never report them.
_SHARED_RULES = frozenset(INPUT_RULES)
# Deliberate exception: a row expression or SQL assertion that cannot be evaluated
# quotes the SQL engine's diagnostic, which quotes the SQL each backend generated.
# Its status, identity and error class must agree; the quoted SQL may differ.
_ENGINE_DIAGNOSTIC = re.compile(r"(could not be evaluated: [A-Za-z ]*Error:).*", re.DOTALL)
_DIAGNOSED = {"row.expression", "sql.assertion"}


def _write(directory: Path, name: str, text: str) -> str:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _columns_scenario(directory: Path) -> dict[str, Any]:
    orders = _write(
        directory,
        "orders.csv",
        "id,amount,code,name,cust,qty,a,b\n"
        "1,10.5,AB,alice,1,1,x,NA\n"
        "1,-3,zz,bob,9,2.5,,y\n"
        "3,,AB,carolinelong,2,3,NA,NA\n",
    )
    customers = _write(directory, "cust.csv", "cid,label\n1,a\n2,b\n")
    return {
        "datasets": {
            "orders": {
                "path": orders,
                "schema": {
                    "columns": {
                        "id": {"dtype": "int"},
                        "amount": {
                            "dtype": "float",
                            "ge": 0,
                            "le": 100,
                            "max_null_fraction": 0.1,
                        },
                        "code": {"dtype": "str", "isin": ["AB", "CD"], "regex": "^[A-Z]+$"},
                        "name": {"dtype": "str", "max_length": 8, "min_length": 4},
                        "cust": {"dtype": "int"},
                        "qty": {"dtype": "float", "integral": True},
                        "a": {"dtype": "str", "nullable": True, "na_values": ["NA"]},
                        "b": {"dtype": "str", "nullable": True, "na_values": ["NA"]},
                        "missing_col": {"dtype": "int"},
                    },
                    "primary_key": ["id"],
                    "na_groups": [["a", "b"]],
                    "min_rows": 5,
                },
                "foreign_keys": [
                    {"columns": ["cust"], "ref_dataset": "cust", "ref_columns": ["cid"]}
                ],
                "checks": [{"check": "expression", "expr": "qty <= 2", "name": "small_qty"}],
            },
            "cust": {
                "path": customers,
                "schema": {"columns": {"cid": {"dtype": "int"}, "label": {"dtype": "str"}}},
            },
        }
    }


def _statistics_scenario(directory: Path) -> dict[str, Any]:
    values = _write(
        directory,
        "values.csv",
        "k,x,y,s,g\n1,1.5,10,Aa,p\n2,2.25,20,aa,p\n3,-1,30,bb,q\n4,100,30,cc,q\n5,,40, dd,r\n",
    )
    return {
        "datasets": {
            "values": {
                "path": values,
                "schema": {
                    "columns": {
                        "k": {"dtype": "int", "unique": True},
                        "x": {
                            "dtype": "float",
                            "nullable": True,
                            "gt": 0,
                            "lt": 50,
                            "max_decimals": 1,
                            "stats": {
                                "min": {"ge": 0},
                                "max": {"le": 10},
                                "mean": {"between": [0, 5]},
                                "median": {"gt": 1},
                                "std": {"lt": 1},
                                "sum": {"le": 50},
                            },
                        },
                        "y": {
                            "dtype": "int",
                            "unique": True,
                            "between": [15, 35],
                            "stats": {"distinct_count": {"ge": 5}},
                            "notin": [40],
                            "conditions": [
                                {"name": "big_when_q", "where": "g = 'q'", "ge": 100},
                                {"where": "g = 'p'", "le": 15},
                                {
                                    "name": "q total",
                                    "where": "g = 'q'",
                                    "stats": {"sum": {"gt": 60}},
                                },
                            ],
                        },
                        "s": {
                            "dtype": "str",
                            "isin": ["aa", "bb", "cc", "dd"],
                            "normalize": ["casefold", "trim"],
                        },
                        "g": {"dtype": "str"},
                    },
                    "primary_key": ["k"],
                },
            }
        }
    }


def _keys_scenario(directory: Path) -> dict[str, Any]:
    rows = _write(
        directory,
        "rows.csv",
        "k1,k2,email,v\n1,a,x@y,1\n1,a,x@y,1\n1,b,,2\n2,a,,3\n2,a,z@y,3\n",
    )
    parents = _write(directory, "parents.csv", "p1,p2\n1,a\n1,b\n9,z\n9,z\n")
    return {
        "datasets": {
            "rows": {
                "path": rows,
                "schema": {
                    "columns": {
                        "k1": {"dtype": "int"},
                        "k2": {"dtype": "str"},
                        "email": {"dtype": "str", "nullable": True},
                        "v": {"dtype": "int"},
                    },
                    "primary_key": ["k1", "k2"],
                    "unique_keys": [["email"], {"columns": ["v"], "nulls": "equal"}],
                    "unique_rows": True,
                    "duplicate_rows_per_key": "all",
                },
                "foreign_keys": [
                    {
                        "columns": ["k1", "k2"],
                        "ref_dataset": "parents",
                        "ref_columns": ["p1", "p2"],
                        "cardinality": "many_to_one",
                    }
                ],
            },
            "parents": {
                "path": parents,
                "schema": {"columns": {"p1": {"dtype": "int"}, "p2": {"dtype": "str"}}},
            },
        }
    }


def _structure_scenario(directory: Path) -> dict[str, Any]:
    shaped = _write(
        directory,
        "shaped.csv",
        "id,when,flag,extra,amount,unit\n"
        "1,2024-01-02,true,e,5,kg\n"
        "x,not-a-date,maybe,e,6,lb\n"
        "3,2024-01-04,false,e,7,\n",
    )
    return {
        "datasets": {
            "shaped": {
                "path": shaped,
                "schema": {
                    "columns": {
                        "id": {"dtype": "int"},
                        "when": {"dtype": "date"},
                        "flag": {"dtype": "bool"},
                        "amount": {"dtype": "int", "unit_column": "unit", "unit": "kg"},
                        "unit": {"dtype": "str", "nullable": True},
                    },
                    "strict": True,
                    "max_rows": 2,
                    "evidence_columns": "all",
                },
                "checks": [{"check": "expression", "expr": "amount > 5", "null_as": "fail"}],
            },
            "absent": {"path": str(directory / "absent.csv"), "required": False},
            "missing": {"path": str(directory / "missing.csv")},
        },
        "checks": [
            {"check": "sql", "name": "few_rows", "sql": "SELECT * FROM shaped WHERE amount > 6"},
            {"check": "sql", "name": "none", "sql": "SELECT * FROM shaped WHERE amount > 100"},
        ],
    }


def _multi_file_scenario(directory: Path) -> dict[str, Any]:
    _write(directory, "parts/a.csv", "id,v\n1,1\n2,-1\n")
    _write(directory, "parts/b.csv", "id,v\n3,-2\n4,4\n")
    return {
        "datasets": {
            "parts": {
                "path": str(directory / "parts" / "*.csv"),
                "schema": {"columns": {"id": {"dtype": "int"}, "v": {"dtype": "int", "ge": 0}}},
            }
        }
    }


def _passing_scenario(directory: Path) -> dict[str, Any]:
    clean = _write(directory, "clean.csv", "id,v,w\n1,1,a\n2,2,b\n")
    return {
        "datasets": {
            "clean": {
                "path": clean,
                "schema": {
                    "columns": {
                        "id": {"dtype": "int"},
                        "v": {"dtype": "int", "ge": 0, "stats": {"mean": {"le": 5}}},
                        "w": {"dtype": "str", "isin": ["a", "b"]},
                    },
                    "primary_key": ["id"],
                },
                "checks": [{"check": "expression", "expr": "v >= 0"}],
            }
        },
        "checks": [{"check": "sql", "name": "empty", "sql": "SELECT * FROM clean WHERE v < 0"}],
    }


def _applicability_scenario(directory: Path) -> dict[str, Any]:
    measures = _write(
        directory,
        "measures.csv",
        "id,score,kind,grade,note\n1,NA,a,10,x\n2,5,a,NA,y\n3,-2,b,30,\n4,NA,b,NA,z\n5,7,,50,w\n",
    )
    return {
        "datasets": {
            "measures": {
                "path": measures,
                "schema": {
                    "columns": {
                        "id": {"dtype": "int"},
                        "score": {
                            "dtype": "float",
                            "na_values": ["NA"],
                            "ge": 0,
                            "stats": {"mean": {"le": 5}},
                        },
                        "kind": {
                            "dtype": "str",
                            "nullable": True,
                            "conditions": [
                                {"where": "id > 3", "isin": ["a"], "min_applicable_rows": 3}
                            ],
                        },
                        "grade": {
                            "dtype": "int",
                            "na_values": ["NA"],
                            "nullable": True,
                            "stats": {"max": {"le": 40}},
                            "unique": True,
                        },
                        "note": {"dtype": "str", "regex": "^[a-z]$", "max_null_fraction": 0.5},
                        "ghost": {"dtype": "int", "required": False},
                    },
                    "primary_key": ["id"],
                    "unique_keys": [{"columns": ["kind"], "where": "id < 5"}],
                },
            }
        },
        "engine": {"max_affected_records": 1},
    }


def _references_scenario(directory: Path) -> dict[str, Any]:
    child = _write(directory, "child.csv", "a,b,v\n1,x,1\n1,,2\n,,3\n2,y,4\n2,y,5\n9,q,6\n")
    parent = _write(directory, "parent.csv", "a,b\n1,x\n2,y\n")
    return {
        "datasets": {
            "child": {
                "path": child,
                "schema": {
                    "columns": {
                        "a": {"dtype": "int", "nullable": True},
                        "b": {"dtype": "str", "nullable": True},
                        "v": {
                            "dtype": "int",
                            "stats": {"sum": {"le": 0}, "std": {"le": 0.1}},
                        },
                    }
                },
                "foreign_keys": [
                    {
                        "columns": ["a", "b"],
                        "ref_dataset": "parent",
                        "ref_columns": ["a", "b"],
                        "match": "full",
                        "cardinality": "one_to_one",
                    },
                    {"columns": ["a"], "ref_dataset": "gone", "ref_columns": ["a"]},
                ],
                "checks": [
                    {"check": "expression", "expr": "nope > 1", "name": "broken"},
                    {"check": "expression", "expr": "v < 6"},
                ],
            },
            "parent": {
                "path": parent,
                "schema": {
                    "columns": {"a": {"dtype": "int"}, "b": {"dtype": "str"}, "z": {"dtype": "int"}}
                },
            },
            "gone": {
                "path": str(directory / "gone.csv"),
                "schema": {"columns": {"a": {"dtype": "int"}}, "primary_key": ["a"]},
                "checks": [{"check": "expression", "expr": "a > 0"}],
            },
        },
        "checks": [
            {"check": "sql", "name": "broken", "sql": "SELECT * FROM nowhere"},
            {"check": "sql", "name": "rows", "sql": "SELECT a FROM child WHERE v > 4"},
        ],
    }


def _undefined_scenario(directory: Path) -> dict[str, Any]:
    empty = _write(directory, "empty.csv", "x,y,u,unit\n,1,1,kg\n,2,2,g\n")
    return {
        "datasets": {
            "empty": {
                "path": empty,
                "schema": {
                    "columns": {
                        "x": {
                            "dtype": "float",
                            "nullable": True,
                            "stats": {"mean": {"ge": 0}, "min": {"ge": 0}},
                        },
                        "y": {
                            "dtype": "int",
                            "stats": {"median": {"ge": 2}, "distinct_count": {"le": 1}},
                        },
                        "u": {"dtype": "int", "unit_column": "units", "unit": "kg"},
                        "unit": {"dtype": "str", "isin": ["kg"]},
                    },
                    "unique_rows": True,
                    "primary_key": ["absent_key"],
                },
            }
        }
    }


def _grouped_scenario(directory: Path) -> dict[str, Any]:
    groups = _write(
        directory,
        "groups.csv",
        "grp,code,active,k\na,07,true,1\na,8,false,1\nb,7,true,\nb,7,true,\nb,9,false,2\n",
    )
    return {
        "datasets": {
            "groups": {
                "path": groups,
                "schema": {
                    "columns": {
                        "code": {"dtype": "int"},
                        "active": {"dtype": "bool"},
                        "k": {"dtype": "int", "nullable": True},
                    },
                    "grouped_values": [
                        {"column": "code", "group_by": ["grp"], "allowed_values": [7, 8]},
                        {
                            "column": "code",
                            "group_by": ["grp"],
                            "allowed_values": [7, 8, 9],
                            "name": "loose",
                        },
                    ],
                    "unique_keys": [
                        {"columns": ["grp", "code"], "where": "active", "name": "active identity"},
                        {"columns": ["k"], "nulls": "forbid"},
                    ],
                    "primary_key": ["grp", "code"],
                    "duplicate_rows_per_key": 1,
                },
            }
        }
    }


def _parquet_scenario(directory: Path) -> dict[str, Any]:
    (directory / "parts").mkdir()
    pq.write_table(pa.table({"id": [1, 2], "v": [1, -1]}), directory / "parts" / "a.parquet")
    pq.write_table(pa.table({"id": [3, 4], "v": [-2, 4]}), directory / "parts" / "b.parquet")
    return {
        "datasets": {
            "parts": {
                "path": str(directory / "parts" / "*.parquet"),
                "schema": {"columns": {"v": {"ge": 0}}, "unique_rows": True},
            }
        }
    }


def _named_checks_scenario(directory: Path) -> dict[str, Any]:
    """Check items with their own severity, name, where and enabled settings."""
    rows = _write(directory, "rows.csv", "id,a,b,kind\n1,1,2,x\n2,5,3,y\n3,4,4,x\n4,,1,y\n")
    return {
        "datasets": {
            "rows": {
                "path": rows,
                "schema": {
                    "columns": {"id": {"dtype": "int"}, "a": {"dtype": "int", "nullable": True}}
                },
                "checks": [
                    {"check": "expression", "expr": "a <= b", "name": "a within b"},
                    {
                        "check": "expression",
                        "expr": "a <= b",
                        "severity": "INFO",
                        "null_as": "fail",
                    },
                    {
                        "check": "expression",
                        "expr": "a < b",
                        "name": "x rows strictly",
                        "where": "kind = 'x'",
                        "severity": "CRITICAL",
                        "description": "only x rows",
                    },
                    {"check": "expression", "expr": "a > 100", "name": "off", "enabled": False},
                ],
            }
        },
        "checks": [
            {"check": "sql", "name": "big a", "sql": "SELECT * FROM rows WHERE a > 4"},
            {
                "check": "sql",
                "name": "big a again",
                "severity": "WARNING",
                "sql": "SELECT * FROM rows WHERE a > 4",
            },
            {"check": "sql", "name": "none", "sql": "SELECT * FROM rows WHERE a > 40"},
        ],
        "run": {"require": ["a within b", "none", "rows"]},
    }


def _across_scenario(directory: Path) -> dict[str, Any]:
    """One dataset copied across files: each copy runs its own contract and checks."""
    sites = {"a": ["1,1", "2,2"], "b": ["1,1", "1,-2"], "c": ["1,1", "2,7"]}
    for site, rows in sites.items():
        _write(directory, f"sites/{site}.csv", "\n".join(["id,x", *rows, ""]))
    return {
        "datasets": {
            "counts": {
                "path": (directory / "sites" / "{site}.csv").as_posix(),
                "across": {"site": ["a", "b", "c"]},
                "schema": {
                    "columns": {"id": {"dtype": "int"}, "x": {"dtype": "int", "ge": 0}},
                    "primary_key": ["id"],
                },
                "checks": [{"check": "expression", "name": "small", "expr": "x < 5"}],
            }
        },
        "checks": [
            {
                "check": "sql",
                "name": "copies agree on ids",
                "sql": 'SELECT id FROM "counts@a" EXCEPT SELECT id FROM "counts@c"',
            }
        ],
        "run": {"require": ["counts@*"]},
    }


def _item_identity_scenario(directory: Path) -> dict[str, Any]:
    """Items sharing a name keep their own tags, policies and requirements."""
    a = _write(directory, "a.csv", "x\n-1\n2\n")
    b = _write(directory, "b.csv", "x\n1\n2\n")
    return {
        "datasets": {
            "a": {
                "path": a,
                "checks": [
                    {"check": "expression", "expr": "x > 0", "tags": ["critical"]},
                    {"check": "expression", "expr": "x > -10", "tags": ["exploratory"]},
                    {"check": "expression", "expr": "x < 10", "where": "x > 0"},
                    {"check": "expression", "name": "x small", "expr": "x < 5"},
                ],
            },
            "b": {
                "path": b,
                "checks": [
                    {"check": "expression", "expr": "x < 10", "where": "x > 0"},
                    {"check": "expression", "name": "x small", "expr": "x < 5", "enabled": False},
                ],
            },
        },
        "checks": [
            {"check": "sql", "name": "no big a", "sql": "SELECT * FROM a WHERE x > 40"},
            {"check": "sql", "name": "no big b", "sql": "SELECT * FROM b WHERE x > 40"},
        ],
        "run": {
            "tags": {"exploratory": {"fail_on": "never"}},
            "require": ["expression on a", "expression on b", "x small", "no big b"],
        },
    }


SCENARIOS = {
    "across": _across_scenario,
    "item_identity": _item_identity_scenario,
    "named_checks": _named_checks_scenario,
    "columns": _columns_scenario,
    "statistics": _statistics_scenario,
    "keys": _keys_scenario,
    "structure": _structure_scenario,
    "multi_file": _multi_file_scenario,
    "passing": _passing_scenario,
    "applicability": _applicability_scenario,
    "references": _references_scenario,
    "undefined": _undefined_scenario,
    "grouped": _grouped_scenario,
    "parquet": _parquet_scenario,
}

_FRAME = {
    "id": [1, 2, 3, 4],
    "x": [None, None, 1.5, float("inf")],
    "g": ["a", "b", "a", None],
    "u": [1, 2, 3, 4],
    "unit": ["kg", "g", "kg", "kg"],
    "k": ["NA", "1", "2", "2"],
}
_FRAME_SCHEMA = {
    "columns": {
        "id": {"dtype": "int"},
        "x": {
            "dtype": "float",
            "nullable": True,
            "stats": {"median": {"ge": 0}},
            "conditions": [
                {"where": "g = 'b'", "ge": 0, "min_applicable_rows": 2, "name": "few"},
                {"where": "g = 'zz'", "stats": {"mean": {"le": 1}}},
            ],
        },
        "g": {"dtype": "str", "nullable": True, "na_values": ["b"], "isin": ["a"]},
        "u": {"dtype": "int", "unit_column": "unit", "allowed_units": ["kg"]},
        "k": {"dtype": "int", "na_values": ["NA"], "unique": True, "stats": {"sum": {"le": 1}}},
    },
    "primary_key": ["id"],
    "unique_keys": [["g"]],
    "na_groups": [["k", "g"]],
}


def _arrow_input() -> Dataset:
    return Dataset(pa.table(_FRAME), schema=_FRAME_SCHEMA)


def _pandas_input() -> Dataset:
    return Dataset(pa.table(_FRAME).to_pandas(), schema=_FRAME_SCHEMA)


def _query_input() -> Dataset:
    return Dataset(
        Query("SELECT * FROM (VALUES (1, -1.5), (2, NULL), (2, 3.0)) t(id, x)"),
        schema={
            "columns": {"x": {"dtype": "float", "ge": 0, "stats": {"median": {"ge": 0}}}},
            "primary_key": ["id"],
        },
    )


INPUTS = {"arrow": _arrow_input, "pandas": _pandas_input, "query": _query_input}


def _normalized(report: Any) -> list[dict[str, Any]]:
    """Input-rule results as JSON values without timings, in a backend-independent order."""
    rows = []
    for result in report.results:
        if result.rule_id not in _SHARED_RULES:
            continue
        row = json.loads(result.model_dump_json())
        for field in _VOLATILE:
            row.pop(field, None)
        if row["status"] == "ERROR" and row["check_id"] in _DIAGNOSED:
            row["message"] = _ENGINE_DIAGNOSTIC.sub(r"\1", row["message"])
        rows.append(row)
    return sorted(rows, key=_order)


def _order(row: dict[str, Any]) -> str:
    return json.dumps(
        [row["rule_id"], row.get("check_id"), row.get("check"), row.get("datasets")],
        default=str,
    )


def _differences(native: list[dict[str, Any]], full: list[dict[str, Any]]) -> list[str]:
    """Readable per-field differences, so a failure points at what diverged."""
    native_keys = [_order(row) for row in native]
    full_keys = [_order(row) for row in full]
    problems = [f"only native: {key}" for key in native_keys if key not in full_keys]
    problems += [f"only polars: {key}" for key in full_keys if key not in native_keys]
    for key, row in zip(native_keys, native, strict=True):
        if key not in full_keys:
            continue
        other = full[full_keys.index(key)]
        for field in sorted(set(row) | set(other)):
            if row.get(field) != other.get(field):
                problems.append(
                    f"{key} {field}: native={row.get(field)!r} polars={other.get(field)!r}"
                )
    return problems


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_backends_report_identical_results(tmp_path, scenario):
    config = SCENARIOS[scenario](tmp_path)
    native_report = validate(config=config, backend="duckdb")
    assert {result.rule_id for result in native_report.results} <= _SHARED_RULES
    native = _normalized(native_report)
    full = _normalized(validate(config=config, backend="polars"))
    problems = _differences(native, full)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("kind", sorted(INPUTS))
def test_backends_report_identical_results_for_supplied_inputs(kind):
    native_report = validate(INPUTS[kind](), backend="duckdb")
    assert {result.rule_id for result in native_report.results} <= _SHARED_RULES
    problems = _differences(
        _normalized(native_report), _normalized(validate(INPUTS[kind](), backend="polars"))
    )
    assert not problems, "\n".join(problems)


def _tagged_scenario(directory: Path) -> dict[str, Any]:
    """Named checks with tags, for run-time selection."""
    config = _named_checks_scenario(directory)
    items = config["datasets"]["rows"]["checks"]
    items[0]["tags"] = ["nightly"]
    items[2]["tags"] = ["nightly", "slow"]
    config["checks"][0]["tags"] = ["nightly"]
    config["run"] = {"require": ["a within b"], "tags": {"slow": {"fail_on": "never"}}}
    return config


SELECTIONS: dict[str, dict[str, Any]] = {
    "tags": {"tags": ["nightly"]},
    "skip_tags": {"skip_tags": ["slow"]},
    "only": {"only": ["a within b", "big a"]},
    "dataset": {"datasets": ["rows"]},
    "family": {"tags": "input", "skip_tags": "nightly"},
}


@pytest.mark.parametrize("selection", sorted(SELECTIONS))
def test_backends_select_the_same_checks(tmp_path, selection):
    config = _tagged_scenario(tmp_path)
    options = SELECTIONS[selection]
    native_report = validate(config=config, backend="duckdb", **options)
    full_report = validate(config=config, backend="polars", **options)
    problems = _differences(_normalized(native_report), _normalized(full_report))
    assert not problems, "\n".join(problems)
    assert native_report.metadata.tag_fail_on == full_report.metadata.tag_fail_on
    assert native_report.verdict() == full_report.verdict()


def test_backends_reject_an_empty_selection(tmp_path):
    from qavf.checks.selection import SelectionError

    config = _tagged_scenario(tmp_path)
    for backend in ("duckdb", "polars"):
        with pytest.raises(SelectionError, match="matches no check"):
            validate(config=config, backend=backend, tags=["nope"])


def test_backends_agree_on_included_files_and_vars(tmp_path):
    rows = _write(tmp_path, "rows.csv", "id,a,b\n1,1,2\n2,5,3\n")
    _write(
        tmp_path,
        "shared.yaml",
        "vars: {limit: 4}\n"
        "datasets:\n"
        f"  rows:\n    path: '{rows}'\n"
        "    schema: {columns: {id: {dtype: int}, a: {dtype: int, le: '${vars.limit}'}}}\n",
    )
    config = _write(
        tmp_path,
        "qa.yaml",
        "include: [shared.yaml]\n"
        "vars: {limit: 3}\n"
        "checks:\n"
        "  - check: sql\n"
        "    name: 'a over ${vars.limit}'\n"
        "    sql: 'SELECT * FROM rows WHERE a > ${vars.limit}'\n",
    )
    native_report = validate(config=config, backend="duckdb")
    full_report = validate(config=config, backend="polars")
    problems = _differences(_normalized(native_report), _normalized(full_report))
    assert not problems, "\n".join(problems)
    assert {r.item_name for r in native_report.results_for("INP-007")} == {"a over 3"}
