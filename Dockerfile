# QAVF command-line image. Build from a verified wheel in dist/:
#   uv build --wheel && docker build -t qavf .
#   docker run --rm -v "$PWD:/work" qavf run -c qavf.yaml
FROM python:3.12-slim

ARG EXTRAS=stats
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 qavf
COPY dist/ /tmp/dist/
RUN set -eu; \
    wheel="$(ls /tmp/dist/qavf-*-py3-none-any.whl)"; \
    pip install --no-cache-dir "${wheel}[${EXTRAS}]"; \
    rm -rf /tmp/dist

USER qavf
# Pre-install the DuckDB extensions QAVF sources use, so runs need no network access.
RUN python -c "import duckdb; c = duckdb.connect(); [c.execute(f'INSTALL {name}') for name in ('httpfs', 'excel', 'postgres', 'mysql', 'sqlite')]"

WORKDIR /work
ENTRYPOINT ["qavf"]
CMD ["--help"]
