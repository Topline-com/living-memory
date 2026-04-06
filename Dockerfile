# ── Stage 1: Server (lightweight, no GPU dependencies) ───────────
FROM python:3.11-slim AS server

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install package (server + client only — no training deps)
COPY pyproject.toml README.md ./
COPY dreamcatcher/ dreamcatcher/
COPY dreamcatcher_client.py config.yaml ./

RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/sessions data/training data/models

EXPOSE 8420

CMD ["dreamcatcher", "serve"]


# ── Stage 2: Training (includes PyTorch + cron for nightly pipeline) ─
FROM python:3.11-slim AS training

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cron && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY dreamcatcher/ dreamcatcher/
COPY dreamcatcher_client.py config.yaml ./
COPY scripts/ scripts/

# Install with training dependencies
RUN pip install --no-cache-dir -e ".[all]"

RUN mkdir -p data/sessions data/training data/models

# Set up nightly cron (3 AM)
RUN echo "0 3 * * * cd /app && dreamcatcher nightly >> /var/log/dreamcatcher.log 2>&1" > /etc/cron.d/dreamcatcher && \
    chmod 0644 /etc/cron.d/dreamcatcher && \
    crontab /etc/cron.d/dreamcatcher && \
    touch /var/log/dreamcatcher.log

CMD ["cron", "-f"]
