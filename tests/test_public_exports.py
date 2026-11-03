"""The lazy top-level exports, ``__all__`` and the type checker's imports agree (A5).

A name missing from the ``TYPE_CHECKING`` imports resolves to ``Any`` for type
checkers through the module ``__getattr__``, silently losing its type.
"""

from __future__ import annotations

import ast
from pathlib import Path

import qavf

INIT = Path(qavf.__file__)


def _type_checking_imports() -> dict[str, str]:
    """Each name imported under ``if TYPE_CHECKING:``, with its module."""
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom) and statement.module:
                    for alias in statement.names:
                        found[alias.asname or alias.name] = statement.module
    return found


def test_exports_all_and_type_checking_inventories_match() -> None:
    typed = _type_checking_imports()
    exports = dict(qavf._EXPORTS)
    assert typed == exports  # the same names, imported from the same modules
    assert set(qavf.__all__) == set(exports) | {"__version__"}
    assert len(qavf.__all__) == len(set(qavf.__all__))


def test_each_typed_import_names_the_runtime_object() -> None:
    for name in qavf._EXPORTS:
        assert getattr(qavf, name) is not None
    assert qavf.CheckIdentity.__module__ == "qavf.reports.models"
