"""Read-query validation and atomic, lexical-aware file argument normalization."""

from __future__ import annotations

import functools
import glob
import json
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from qavf.sources.base import SourceError
from qavf.sources.errors import redacted_error


def sql_literal(value: Any) -> str:
    """Render a Python value as a DuckDB SQL literal (for reader options)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Mapping):
        items = ", ".join(f"{sql_literal(str(k))}: {sql_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(sql_literal(v) for v in value) + "]"
    return "'" + str(value).replace("'", "''") + "'"


_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'")
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_FILE_NAME = re.compile(
    r"\.(?:csv|tsv|txt|parquet|pq|json|ndjson|jsonl|xlsx)"
    r"(?:\.(?:gz|gzip|zst|zstd))?(?:[?#].*)?$",
    re.IGNORECASE,
)


def _base_tables(node: Any, found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("type") == "BASE_TABLE":
            found.append(node)
        for value in node.values():
            _base_tables(value, found)
    elif isinstance(node, list):
        for value in node:
            _base_tables(value, found)
    return found


_PARSER = threading.local()


def _parser() -> duckdb.DuckDBPyConnection:
    """A per-thread connection used only to parse SQL (opening one costs ~40 ms)."""
    con = getattr(_PARSER, "con", None)
    if con is None:
        con = _PARSER.con = duckdb.connect()
    return con


def sql_catalogs(query: str) -> set[str]:
    """See :func:`_sql_catalogs` (cached: the same query is parsed once)."""
    return set(_sql_catalogs(query))


@functools.lru_cache(maxsize=1024)
def _sql_catalogs(query: str) -> frozenset[str]:
    """Names ``query`` uses as a database (catalog) qualifier, e.g. ``w`` in ``w.items``.

    Uses DuckDB's parser, so aliases (``SELECT w.x FROM ... AS w``), comments,
    string literals and CTE names are not mistaken for databases, and quoted
    (``"w".items``) or differently cased (``W.items``) references are found. In a
    two-part name the qualifier may be a schema rather than a database; callers
    match the result against their configured connection names. If DuckDB
    cannot parse the query, qualifiers outside literals and comments are used.
    """
    try:
        row = _parser().execute(f"SELECT json_serialize_sql({sql_literal(query)})").fetchone()
        tree = json.loads(row[0]) if row else {"error": True}
    except (duckdb.Error, TypeError, ValueError):
        tree = {"error": True}
    if tree.get("error"):
        stripped = _SQL_COMMENT.sub(" ", _SQL_LITERAL.sub("''", query))
        return frozenset(
            (quoted or bare)
            for quoted, bare in re.findall(r'"([^"]+)"\s*\.|\b([A-Za-z_]\w*)\s*\.', stripped)
        )
    names = set()
    for table in _base_tables(tree, []):
        qualifier = table.get("catalog_name") or table.get("schema_name")
        if qualifier:
            names.add(str(qualifier))
    return frozenset(names)


def referenced_connections(query: str, connections: Iterable[str]) -> list[str]:
    """The configured ``connections`` that ``query`` reads from (case-insensitive)."""
    used = {name.lower() for name in sql_catalogs(query)}
    return [name for name in connections if name.lower() in used]


def quote_identifier(name: str) -> str:
    """Double-quote an SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


#: DuckDB table functions whose first argument is a file path, glob or list of them.
_FILE_FUNCTIONS = (
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "parquet_metadata",
    "parquet_schema",
    "read_json",
    "read_json_auto",
    "read_json_objects",
    "read_ndjson",
    "read_ndjson_auto",
    "read_ndjson_objects",
    "read_xlsx",
    "read_text",
    "read_blob",
    "glob",
    "sniff_csv",
)


@dataclass(frozen=True, slots=True)
class _SqlToken:
    kind: str
    text: str
    start: int
    end: int


def _tokens(sql: str) -> list[_SqlToken]:
    """Lex SQL while keeping comments and quoted data out of structural matching."""
    found: list[_SqlToken] = []
    i, length = 0, len(sql)
    while i < length:
        start = i
        char = sql[i]
        if char.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = length if end < 0 else end + 1
            continue
        if sql.startswith("/*", i):
            depth, i = 1, i + 2
            while i < length and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            continue
        if char in ("e", "E") and i + 1 < length and sql[i + 1] == "'":
            i += 2
            while i < length:
                if sql[i] == "\\":
                    i = min(length, i + 2)
                elif sql[i] == "'":
                    if i + 1 < length and sql[i + 1] == "'":
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
            found.append(_SqlToken("string", sql[start:i], start, i))
            continue
        if char in ("'", '"'):
            i += 1
            while i < length:
                if sql[i] == char:
                    if i + 1 < length and sql[i + 1] == char:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            found.append(_SqlToken("string" if char == "'" else "quoted", sql[start:i], start, i))
            continue
        if char == "$":
            delimiter = re.match(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$", sql[i:])
            if delimiter:
                mark = delimiter.group()
                end = sql.find(mark, i + len(mark))
                i = length if end < 0 else end + len(mark)
                found.append(_SqlToken("dollar", sql[start:i], start, i))
                continue
        if char.isalpha() or char == "_":
            i += 1
            while i < length and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            found.append(_SqlToken("word", sql[start:i], start, i))
            continue
        i += 1
        found.append(_SqlToken("symbol", char, start, i))
    return found


def _absolute_path(value: Any, base_dir: Path, *, glob_reader: bool = True) -> str | list[str]:
    """Normalize only concrete paths; never evaluate a caller's SQL expression."""
    if isinstance(value, list | tuple):
        paths = [_absolute_path(item, base_dir, glob_reader=glob_reader) for item in value]
        if not paths or any(not isinstance(item, str) for item in paths):
            raise SourceError("SQL file lists must contain one or more scalar paths")
        return paths  # type: ignore[return-value]
    if not isinstance(value, str | Path) or not str(value):
        raise SourceError("SQL file arguments must be nonempty paths or lists of paths")
    path = str(value)
    if "://" in path:
        return path
    resolved = Path(path) if Path(path).is_absolute() else (base_dir / path).resolve()
    # An existing path is literal even if its name contains glob metacharacters.
    # Non-existing patterns retain deliberate glob semantics.
    return glob.escape(str(resolved)) if glob_reader and resolved.is_file() else str(resolved)


def _query_table_path(value: Any, base_dir: Path) -> Any:
    """Keep SQL table names; refuse relative filename replacement scans."""
    if isinstance(value, list | tuple):
        return [_query_table_path(item, base_dir) for item in value]
    if isinstance(value, str) and _FILE_NAME.search(value):
        if "://" in value or Path(value).is_absolute():
            return _absolute_path(value, base_dir)
        raise SourceError(
            "ambiguous filename in query_table(); use an explicit file reader such as "
            "read_csv('relative.csv'), resolved against the configuration directory"
        )
    return value


def _literal_paths(node: Mapping[str, Any]) -> str | list[str]:
    """Accept a complete constant argument; reject expressions before rewriting."""
    if node.get("class") == "CONSTANT":
        value = node.get("value", {})
        if value.get("type", {}).get("id") == "VARCHAR" and not value.get("is_null"):
            return str(value["value"])
    if node.get("class") == "FUNCTION" and node.get("function_name") == "list_value":
        values = [_literal_paths(child) for child in node.get("children", ())]
        if values and all(isinstance(value, str) for value in values):
            return values  # type: ignore[return-value]
    raise SourceError(
        "dynamic SQL file paths are unsupported; use literal paths, literal lists, "
        "or one named bound path parameter"
    )


def _argument_end(tokens: list[_SqlToken], start: int) -> int:
    """Locate the complete first argument, including any expression after a list."""
    depth = 0
    for position in range(start, len(tokens)):
        text = tokens[position].text
        if depth == 0 and text in (",", ")"):
            return position
        if text in ("(", "["):
            depth += 1
        elif text in (")", "]"):
            depth -= 1
    raise SourceError("unterminated SQL file argument")


def normalize_sql_paths(
    sql: str,
    base_dir: Path,
    parameters: Mapping[str, Any] | None = None,
    *,
    known_tables: Iterable[str] = (),
) -> tuple[str, dict[str, Any]]:
    """Resolve complete file arguments while retaining named parameter bindings.

    Comments, escaped quotes, bracket characters inside names and dollar-quoted
    values cannot change parsing. Unsupported dynamic expressions are rejected
    atomically; no part of an expression is left to resolve against process CWD.
    """
    bound = dict(parameters or {})
    table_names = frozenset(name.casefold() for name in known_tables)
    parameter_names = {name.casefold(): name for name in bound}
    if len(parameter_names) != len(bound):
        raise SourceError("SQL parameter names must not repeat with different casing")
    tokens = _tokens(sql)
    # DuckDB identifies actual table references; the lexer supplies exact spans.
    # This keeps EXTRACT(... FROM 'date') and other expression grammar untouched.
    try:
        row = _parser().execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
        tree = json.loads(row[0]) if row else {"error": True}
    except (duckdb.Error, TypeError, ValueError) as exc:
        raise redacted_error(exc) from None
    if tree.get("error"):
        # Dynamic PIVOT expands to an internal CREATE ENUM plus SELECT, which
        # DuckDB's JSON SELECT serializer cannot represent. A direct table pivot
        # contains no file location to resolve; nested/file pivots remain explicit.
        if (
            len(tokens) > 1
            and tokens[0].text.lower() == "pivot"
            and tokens[1].kind in ("word", "quoted")
            and tokens[1].text.lower() not in _FILE_FUNCTIONS
            and not any(t.kind == "word" and t.text.lower() in _FILE_FUNCTIONS for t in tokens)
            and not any(
                t.text.lower() in ("from", "join") and tokens[i + 1].kind == "string"
                for i, t in enumerate(tokens[:-1])
            )
        ):
            if tokens[1].kind == "quoted" and _FILE_NAME.search(tokens[1].text[1:-1]):
                raise SourceError(
                    "ambiguous quoted filename in PIVOT; use an explicit file reader "
                    "with a configuration-relative path"
                )
            return sql, bound
        raise SourceError("could not parse SQL while resolving file paths")
    locations: dict[int, tuple[dict[str, Any], str]] = {}
    replacements_at: dict[int, str] = {}
    ambiguous_at: set[int] = set()
    parameter_counts: dict[str, int] = {}

    def walk(node: Any, ctes: frozenset[str] = frozenset()) -> None:
        if isinstance(node, dict):
            if node.get("class") == "PARAMETER":
                identifier = str(node["identifier"]).casefold()
                parameter_counts[identifier] = parameter_counts.get(identifier, 0) + 1
            entries = node.get("cte_map", {}).get("map", ())
            visible = ctes | frozenset(str(item["key"]).casefold() for item in entries)
            if node.get("type") == "RECURSIVE_CTE_NODE":
                visible = visible | frozenset((str(node.get("cte_name", "")).casefold(),))
            if node.get("type") == "TABLE_FUNCTION":
                function = node.get("function", {})
                name = function.get("function_name", "").lower()
                if name in (*_FILE_FUNCTIONS, "query_table"):
                    arguments = function.get("children", ())
                    if arguments:
                        locations[node.get("query_location", -1)] = (arguments[0], name)
            elif node.get("type") == "BASE_TABLE":
                name = node.get("table_name", "")
                location = node.get("query_location", -1)
                replacements_at[location] = name
                qualified = ".".join(
                    value
                    for value in (node.get("catalog_name"), node.get("schema_name"), name)
                    if value
                ).casefold()
                known = qualified in table_names or (
                    not node.get("catalog_name")
                    and node.get("schema_name", "").lower() in ("", "main")
                    and name.casefold() in table_names
                )
                if _FILE_NAME.search(name) and name.casefold() not in visible and not known:
                    ambiguous_at.add(location)
            # A nonrecursive CTE name does not hide a physical table inside its
            # own definition. Otherwise a self-named CTE could bypass the guard.
            prior = ctes
            for entry in entries:
                walk(entry["value"], prior)
                prior = prior | frozenset((str(entry["key"]).casefold(),))
            for key, value in node.items():
                if key != "cte_map":
                    walk(value, visible)
        elif isinstance(node, list):
            for value in node:
                walk(value, ctes)

    walk(tree)
    if not locations and not replacements_at:
        return sql, bound
    byte_locations: dict[int, int] = {}
    previous = offset = 0
    for token in tokens:
        offset += len(sql[previous : token.start].encode("utf-8"))
        byte_locations[token.start] = offset
        previous = token.start
    replacements: dict[int, tuple[int, str]] = {}
    path_parameters: dict[str, list[tuple[int, int, Any]]] = {}
    for index, token in enumerate(tokens):
        location = byte_locations[token.start]
        if location in replacements_at and token.kind in ("string", "dollar"):
            replacements[token.start] = (
                token.end,
                sql_literal(_absolute_path(replacements_at[location], base_dir)),
            )
        elif location in ambiguous_at:
            raise SourceError(
                "ambiguous quoted filename/table identifier; use an explicit file reader "
                "such as read_csv('relative.csv'), resolved against the configuration directory"
            )
        opening = index + 1
        while opening + 1 < len(tokens) and tokens[opening].text == ".":
            opening += 2
        if location in locations and opening < len(tokens) and tokens[opening].text == "(":
            argument = opening + 1
            if argument >= len(tokens):
                continue
            end = _argument_end(tokens, argument)
            expression, function = locations[location]

            def resolve(value: Any, reader: str = function) -> Any:
                if reader == "query_table":
                    return _query_table_path(value, base_dir)
                return _absolute_path(value, base_dir, glob_reader=reader != "read_xlsx")

            if expression.get("class") == "PARAMETER":
                name = str(expression["identifier"])
                key = parameter_names.get(name.casefold())
                if key is None:
                    raise SourceError(f"SQL file parameter {name!r} has no bound value")
                path_parameters.setdefault(key, []).append(
                    (tokens[argument].start, tokens[end - 1].end, resolve(bound[key]))
                )
            else:
                value = resolve(_literal_paths(expression))
                replacements[tokens[argument].start] = (tokens[end - 1].end, sql_literal(value))
    internal_index = 0
    reserved = set(parameter_names) | set(parameter_counts)
    positional_index = max((int(name) for name in reserved if name.isdecimal()), default=0) + 1
    for key, occurrences in path_parameters.items():
        other_uses = parameter_counts.get(key.casefold(), 0) > len(occurrences)
        same_resolution = all(value == occurrences[0][2] for _, _, value in occurrences)
        if not other_uses and same_resolution:
            bound[key] = occurrences[0][2]
            continue
        for start, end, value in occurrences:
            if key.isdecimal():
                internal = str(positional_index)
                positional_index += 1
            else:
                while (internal := f"__qavf_path_{internal_index}").casefold() in reserved:
                    internal_index += 1
                internal_index += 1
            reserved.add(internal.casefold())
            bound[internal] = value
            replacements[start] = end, f"${internal}"
        if not other_uses:
            bound.pop(key)
    fragments: list[str] = []
    previous = 0
    for start, (end, value) in sorted(replacements.items()):
        fragments.extend((sql[previous:start], value))
        previous = end
    fragments.append(sql[previous:])
    return "".join(fragments), bound


def resolve_sql_paths(sql: str, base_dir: Path, *, known_tables: Iterable[str] = ()) -> str:
    """Normalize SQL with literal file paths; parameterized callers use normalize_sql_paths."""
    return normalize_sql_paths(sql, base_dir, known_tables=known_tables)[0]


def normalize_single_query(sql: str) -> str:
    """Remove a statement terminator, retaining quoted semicolons and comments."""
    tokens = _tokens(sql)
    terminators = [token for token in tokens if token.kind == "symbol" and token.text == ";"]
    if not terminators:
        return sql
    if len(terminators) != 1 or terminators[0] is not tokens[-1]:
        raise SourceError("a source must contain exactly one SQL query")
    token = terminators[0]
    return sql[: token.start] + sql[token.end :]


def validate_query(sql: str, *, connection: Any | None = None) -> str:
    """Normalize a single result-producing SELECT, rejecting modifying statements."""
    sql = normalize_single_query(sql)
    try:
        statements = (connection if connection is not None else _parser()).extract_statements(sql)
    except duckdb.Error as exc:
        raise redacted_error(exc) from None
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise SourceError("a source must contain exactly one SELECT query")
    return sql
