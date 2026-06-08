# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+docker

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY example/default_configs ./default_configs
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

RUN mkdir -p \
        /app/data \
        /app/models \
        /app/results \
        /app/runs \
    # Remove the next two lines if you want the container to run as root (dev mode).
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

# Remove this line if you want the container to run as root (dev mode)..
USER appuser

RUN python -c "from autorec.runtime import configure_autorec_runtime; configure_autorec_runtime(thread_count=1, warmup_autoeis=True)"

# Keep these defaults for CLI usage: docker run --rm autorec:latest train --config ...
ENTRYPOINT ["autorec"]
CMD ["--help"]

# For an interactive Bash shell:
# 1. Comment out ENTRYPOINT and CMD above
# 2. Uncomment the line below
# CMD ["/bin/bash"]
