FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CIPHERLENS_TORCH_THREADS=2

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser assets ./assets
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser .streamlit ./.streamlit

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from pathlib import Path; import urllib.request; assert Path('/app/models/captcha_crnn.pt').is_file(); urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
