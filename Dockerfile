FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORGE_DATA_DIR=/var/lib/forge

WORKDIR /app

RUN useradd --create-home --uid 10001 forge \
    && mkdir -p /var/lib/forge \
    && chown forge:forge /var/lib/forge

COPY . /app
RUN python -m pip install --upgrade pip \
    && python -m pip install .

USER forge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=2)"

CMD ["forge-web"]
