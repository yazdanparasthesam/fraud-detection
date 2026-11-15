FROM python:3.12-slim

# uv does the installing: faster than pip and, with uv.lock, reproducible.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so the layer is cached across code edits
COPY requirements-dev.txt .
RUN uv venv "$VIRTUAL_ENV" && uv pip install --no-cache -r requirements-dev.txt

# Application code. predict.py imports inference.py and reasons.py, and
# inference.py imports reasons.py - all must be present or the import fails
# when the container starts.
COPY train.py predict.py vectorize.py reasons.py inference.py ./

# The model artefact is produced by `python train.py`. It is intentionally NOT
# baked into the image: the 471 MB dataset is not committed to the repo either.
# Build the image, then either
#   (a) mount the trained model:  docker run -v $(pwd)/models:/app/models ...
#   (b) or train inside a container and commit it.
# See README "Run with Docker" for the exact commands.
RUN mkdir -p /app/models

EXPOSE 9696

# The container expects the model to be present. If it is missing, predict.py
# starts, fails fast with a clear message, and the container exits non-zero
# instead of serving wrong scores.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:9696/health').status==200 else 1)"

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
