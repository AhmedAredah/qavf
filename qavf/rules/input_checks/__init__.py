"""Configuration-driven input rules, grouped by validation responsibility.

Concrete public rule modules are discovered recursively by RuleRegistry.
Execution helpers are private and do not register additional rules.
"""

from ._datasets import DatasetSelection
from .availability import RequiredDatasetsRule, SchemaConformanceRule
from .columns import ColumnCheckRule
from .expressions import ExpressionRule, SqlAssertionRule
from .keys import DuplicateKeyRule
from .references import ReferentialIntegrityRule

__all__ = [
    "ColumnCheckRule",
    "DatasetSelection",
    "DuplicateKeyRule",
    "ExpressionRule",
    "ReferentialIntegrityRule",
    "RequiredDatasetsRule",
    "SchemaConformanceRule",
    "SqlAssertionRule",
]
