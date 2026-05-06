FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY privateye/ privateye/
COPY scripts/ scripts/
COPY tests/ tests/

# Data and log dirs
RUN mkdir -p data/historical logs

EXPOSE 8081

CMD ["python", "-m", "privateye.main", "--mode", "paper"]
