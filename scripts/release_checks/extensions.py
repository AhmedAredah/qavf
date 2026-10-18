"""Provision signed DuckDB extensions without depending on HTTP bootstrap access."""

from __future__ import annotations

import gzip
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


def bootstrap_httpfs(connection: Any) -> None:
    """Download the matching signed binary with Python TLS, then use normal INSTALL.

    Older DuckDB versions need httpfs to download another extension over HTTPS,
    but auto-install httpfs over HTTP. Local installation breaks that dependency
    cycle without changing DuckDB's signature or compatibility checks.
    """
    version = connection.execute("SELECT version()").fetchone()[0]
    platform = connection.execute("PRAGMA platform").fetchone()[0]
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version):
        raise RuntimeError(f"Signed extension bootstrap requires a release DuckDB build: {version}")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", platform):
        raise RuntimeError(f"Unsupported DuckDB extension platform identifier: {platform}")
    url = f"https://extensions.duckdb.org/{version}/{platform}/httpfs.duckdb_extension.gz"
    with tempfile.TemporaryDirectory(prefix="qavf-extension-") as directory:
        target = Path(directory) / "httpfs.duckdb_extension"
        # Default urllib HTTPS certificate/hostname verification remains enabled.
        with urllib.request.urlopen(url, timeout=30) as response:
            if not response.geturl().startswith("https://"):
                raise RuntimeError("Extension repository redirected outside HTTPS")
            with gzip.GzipFile(fileobj=response) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=64 * 1024)
        path = str(target).replace("'", "''")
        connection.execute(f"INSTALL '{path}'")
        connection.execute("LOAD httpfs")
    print(f"Bootstrapped signed DuckDB {version} httpfs for {platform} through HTTPS", flush=True)


def install_extensions(live: bool = False) -> None:
    import duckdb

    names = ("excel", "sqlite") + (("postgres", "httpfs") if live else ())
    with duckdb.connect() as connection:
        for name in names:
            canonical = {"sqlite": "sqlite_scanner", "postgres": "postgres_scanner"}.get(name, name)
            installed = connection.execute(
                "SELECT installed FROM duckdb_extensions() WHERE extension_name = ?", [canonical]
            ).fetchone()
            if installed and installed[0]:
                # LOAD verifies the existing binary; no repository-origin change is needed.
                connection.execute(f"LOAD {name}")
                print(
                    f"Verified installed DuckDB {duckdb.__version__} extension: {name}", flush=True
                )
                continue
            statement = f"INSTALL {name} FROM 'https://extensions.duckdb.org'"
            try:
                connection.execute(statement)
            except duckdb.Error as exc:
                # Only the known HTTPS transport bootstrap failure has a fallback.
                if "required extension 'httpfs'" not in str(exc):
                    raise
                bootstrap_httpfs(connection)
                connection.execute(statement)
            connection.execute(f"LOAD {name}")
            print(f"Verified DuckDB {duckdb.__version__} extension: {name}", flush=True)
