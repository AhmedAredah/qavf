"""Pause a real first pandas import after sys.modules publication."""

import importlib.abc
import importlib.machinery
import json
import sys
import threading

import pyarrow as pa

from qavf.sources.frames import supports_frame

table = pa.Table.from_batches([], schema=pa.schema([("x", pa.int64())]))
if "pandas" in sys.modules:
    raise RuntimeError("pandas was already imported; probe needs a fresh import")
entered = threading.Event()
release = threading.Event()
errors = []


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "pandas":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return original.create_module(spec)

            def exec_module(self, module):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("probe barrier expired")
                original.exec_module(module)

        spec.loader = Loader()
        return spec


def importing():
    try:
        importlib.import_module("pandas")
    except BaseException as exc:
        errors.append(repr(exc))


finder = Finder()
before = supports_frame(table)
sys.meta_path.insert(0, finder)
thread = threading.Thread(target=importing)
thread.start()
try:
    if not entered.wait(10):
        raise TimeoutError("import did not start")
    try:
        during = {"accepted": supports_frame(table)}
    except Exception as exc:
        during = {"error": type(exc).__name__, "message": str(exc)}
finally:
    release.set()
    thread.join(20)
    sys.meta_path.remove(finder)
print(
    json.dumps(
        {
            "accepted_before": before,
            "during_import": during,
            "accepted_after": supports_frame(table),
            "import_errors": errors,
        },
        indent=2,
    )
)
