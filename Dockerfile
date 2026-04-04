FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cron && \
    rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md ./
COPY dreamcatcher/ dreamcatcher/
COPY dreamcatcher_client.py config.yaml ./
COPY scripts/ scripts/

# Install with training dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Create data directories
RUN mkdir -p data/sessions data/training data/models

# Set up nightly cron (3 AM)
RUN echo "0 3 * * * cd /app && dreamcatcher nightly >> /var/log/dreamcatcher.log 2>&1" > /etc/cron.d/dreamcatcher && \
    chmod 0644 /etc/cron.d/dreamcatcher && \
    crontab /etc/cron.d/dreamcatcher && \
    touch /var/log/dreamcatcher.log

EXPOSE 8420

# Default: run the inference server
CMD ["dreamcatcher", "serve"]
