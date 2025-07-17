# FormFinder Docker Configuration
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FORMFINDER_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r formfinder && useradd -r -g formfinder formfinder

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=formfinder:formfinder . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/raw /app/processed_data && \
    chown -R formfinder:formfinder /app

# Switch to non-root user
USER formfinder

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD formfinder health-check || exit 1

# Expose port (if web interface is added in future)
EXPOSE 8080

# Default command
CMD ["formfinder", "run"]

# Development stage
FROM production as development

# Switch back to root for development setup
USER root

# Install development dependencies
RUN pip install -e .[dev,postgresql,sms,docs,monitoring]

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to formfinder user
USER formfinder

# Development command (interactive shell)
CMD ["/bin/bash"]