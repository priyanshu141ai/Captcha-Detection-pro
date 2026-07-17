ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

FROM ${PYTHON_IMAGE} AS builder

ARG TORCH_VERSION=2.12.1
ARG PIP_VERSION=26.1.2
ARG SETUPTOOLS_VERSION=83.0.0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build

COPY requirements.txt pyproject.toml README.md ./
RUN python -m venv --copies /opt/venv \
    && python -m pip install --upgrade \
        "pip==${PIP_VERSION}" \
        "setuptools==${SETUPTOOLS_VERSION}" \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" \
    && python -m pip install --requirement requirements.txt

COPY src ./src
RUN python -m pip install --no-build-isolation --no-deps . \
    && python -m pip check

FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    CIPHERLENS_TORCH_THREADS=2

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 app.py ./
COPY --chown=10001:10001 assets ./assets
COPY --chown=10001:10001 configs ./configs
COPY --chown=10001:10001 models ./models
COPY --chown=10001:10001 .streamlit ./.streamlit

USER 10001:10001

EXPOSE 8000 8501
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
