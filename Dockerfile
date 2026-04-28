FROM python:3.12-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies if needed (none required for current setup, but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Upgrade pip and install the project with development dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Default command to run the simulation
CMD ["python", "scripts/run_scenario.py"]
