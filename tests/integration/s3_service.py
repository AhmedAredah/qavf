"""Unique test-owned S3 bucket with deterministic files and unconditional cleanup."""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from qavf.sources.sql import sql_literal

from .cleanup import cleanup_owned
from .contracts import corpus
from .policy import IntegrationUnavailable, dependency, endpoint_parts, extension, settings


@dataclass(frozen=True)
class S3Corpus:
    bucket: str
    endpoint: str
    client: Any = field(repr=False)
    secret_sql: str = field(repr=False)

    def key(self, clean: bool) -> str:
        return "clean.parquet" if clean else "invalid.parquet"

    def uri(self, clean: bool) -> str:
        return f"s3://{self.bucket}/{self.key(clean)}"

    def public_url(self, key: str) -> str:
        return f"{self.endpoint.rstrip('/')}/{self.bucket}/{key}"


@contextmanager
def s3_corpus() -> Iterator[S3Corpus]:
    config = settings(
        os.environ,
        (
            "QAVF_S3_ENDPOINT",
            "QAVF_S3_ACCESS_KEY",
            "QAVF_S3_SECRET_KEY",
        ),
    )
    endpoint, use_ssl = endpoint_parts(config["QAVF_S3_ENDPOINT"])
    boto3 = dependency("boto3")
    botocore = dependency("botocore.config")
    extension("httpfs")
    region = os.environ.get("QAVF_S3_REGION", "us-east-1")
    client = boto3.client(
        "s3",
        endpoint_url=config["QAVF_S3_ENDPOINT"],
        region_name=region,
        aws_access_key_id=config["QAVF_S3_ACCESS_KEY"],
        aws_secret_access_key=config["QAVF_S3_SECRET_KEY"],
        config=botocore.Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=5,
            retries={"max_attempts": 0},
            s3={"addressing_style": "path"},
        ),
    )
    bucket = "qavf-integration-" + uuid.uuid4().hex
    created = False
    keys: list[str] = []
    try:
        try:
            options = (
                {}
                if region == "us-east-1"
                else {"CreateBucketConfiguration": {"LocationConstraint": region}}
            )
            client.create_bucket(Bucket=bucket, **options)
            created = True
            client.put_bucket_policy(
                Bucket=bucket,
                Policy=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": "*",
                                "Action": "s3:GetObject",
                                "Resource": f"arn:aws:s3:::{bucket}/*",
                            }
                        ],
                    }
                ),
            )
            for clean in (True, False):
                key = "clean.parquet" if clean else "invalid.parquet"
                buffer = io.BytesIO()
                corpus(clean).write_parquet(buffer)
                # Record ownership before upload so an ambiguous response still cleans up.
                keys.append(key)
                client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        except Exception as exc:
            raise IntegrationUnavailable(
                f"required S3 service or fixture setup unavailable ({type(exc).__name__})"
            ) from None
        secret = (
            "CREATE SECRET qavf_integration (TYPE S3, PROVIDER CONFIG, "
            f"KEY_ID {sql_literal(config['QAVF_S3_ACCESS_KEY'])}, "
            f"SECRET {sql_literal(config['QAVF_S3_SECRET_KEY'])}, "
            f"REGION {sql_literal(region)}, ENDPOINT {sql_literal(endpoint)}, "
            f"URL_STYLE 'path', USE_SSL {sql_literal(use_ssl)})"
        )
        yield S3Corpus(bucket, config["QAVF_S3_ENDPOINT"], client, secret)
    finally:
        actions = []
        if created:
            actions.extend(
                lambda key=key: client.delete_object(Bucket=bucket, Key=key) for key in keys
            )
            actions.append(lambda: client.delete_bucket(Bucket=bucket))
        actions.append(client.close)
        cleanup_owned("S3", actions, sys.exception())
