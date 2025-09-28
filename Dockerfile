# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml first for better Docker layer caching
COPY pyproject.toml ./

# Copy source code
COPY src/ ./src/

# Install the project in development mode
RUN pip install --no-cache-dir -e .

# Set default command to show help
CMD ["qbt", "--help"]