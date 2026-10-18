"""Unique test-owned Azure Blob container with deterministic files and unconditional cleanup."""

from __future__ import annotations

import io
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .cleanup import cleanup_owned
from .contracts import corpus
from .policy import IntegrationUnavailable, dependency, extension, settings


@dataclass(frozen=True)
class AzureCorpus:
    container: str
    connection_string: str = field(repr=False)
    client: Any = field(repr=False)

    def uri(self, clean: bool) -> str:
        return f"az://{self.container}/{'clean' if clean else 'invalid'}.parquet"


@contextmanager
def azure_corpus() -> Iterator[AzureCorpus]:
    config = settings(os.environ, ("QAVF_AZURE_CONNECTION_STRING",))
    blob = dependency("azure.storage.blob")
    extension("azure")
    connection_string = config["QAVF_AZURE_CONNECTION_STRING"]
    service = blob.BlobServiceClient.from_connection_string(connection_string)
    container = "qavf-integration-" + uuid.uuid4().hex
    created = False
    try:
        try:
            client = service.create_container(container)
            created = True
            for clean in (True, False):
                buffer = io.BytesIO()
                corpus(clean).write_parquet(buffer)
                client.upload_blob(f"{'clean' if clean else 'invalid'}.parquet", buffer.getvalue())
        except Exception as exc:
            raise IntegrationUnavailable(
                f"required Azure service or fixture setup unavailable ({type(exc).__name__})"
            ) from None
        yield AzureCorpus(container, connection_string, client)
    finally:
        actions = []
        if created:
            actions.append(lambda: service.delete_container(container))
        actions.append(service.close)
        cleanup_owned("Azure", actions, sys.exception())
