# Gateway image — slim Python with our package and runtime deps.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps: curl is needed for the healthcheck; build-essential lets
# pyarrow / argon2-cffi / pynacl wheels build cleanly when binaries are
# missing for an arch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first to maximize layer cache.
COPY requirements.txt /app/provider/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/provider/requirements.txt

# Copy the package itself (build context is the provider/ folder).
COPY . /app/provider

# Where the SQLite control DB and Lance datasets live. Mounted as volumes
# from compose.yml so the data outlives container rebuilds.
RUN mkdir -p /app/data /app/data/lance

EXPOSE 8088

# Default command — the gateway listens on 0.0.0.0:8088.
CMD ["python", "-m", "provider", "--host", "0.0.0.0", "--port", "8088", "--log-level", "info"]
