# Multi-stage Docker build for Aura Render video generation platform
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Configure Huawei Cloud apt mirror for faster downloads in China
RUN sed -i 's|http://deb.debian.org|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's|http://deb.debian.org|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's|http://security.debian.org|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list

# Install system dependencies (FFmpeg已移除，使用阿里云IMS进行视频处理)
RUN apt-get update && apt-get install -y \
    # Essential system packages
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    # Image processing libraries (保留，用于图片预处理)
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Minimal OpenCV dependencies (even headless version needs these)
    libglvnd0 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    # Audio processing libraries
    libsndfile1 \
    # Database support
    libsqlite3-dev \
    # Other essential libraries
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
# Use Huawei Cloud PyPI mirror for faster downloads in China
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple/

# Development stage with additional tools
FROM base AS development

# Install development dependencies (apt mirror already configured in base stage)
RUN apt-get update && apt-get install -y \
    # Development tools
    vim \
    htop \
    tree \
    # Debugging tools
    gdb \
    strace \
    # Network tools
    netcat-openbsd \
    telnet \
    # Additional Python tools
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages (Use Huawei Cloud PyPI mirror)
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple/

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/output /app/logs /app/cache

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 5555

# Development command with hot reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]

# Production stage - optimized for production deployment
FROM base AS production

# Install production-only dependencies
RUN apt-get update && apt-get install -y \
    # Production monitoring tools
    htop \
    # Minimal tools for container health
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application code (excluding dev files via .dockerignore)
COPY . .

# Install only production Python dependencies (Use Huawei Cloud PyPI mirror)
RUN pip install --no-cache-dir gunicorn uvicorn[standard] -i https://mirrors.huaweicloud.com/repository/pypi/simple/

# Create necessary directories with proper permissions
RUN mkdir -p /app/temp /app/output /app/logs /app/cache /app/uploads \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app

# Create volume mount points
VOLUME ["/app/temp", "/app/output", "/app/logs", "/app/uploads"]

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command with Gunicorn
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "--access-logfile", "-", "--error-logfile", "-"]

# Worker stage for Celery workers
FROM base AS worker

# Install worker-specific dependencies (if any)
RUN apt-get update && apt-get install -y \
    # Additional tools for workers
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/output /app/logs /app/cache \
    && chown -R appuser:appuser /app

# Switch to app user  
USER appuser

# Worker command
CMD ["celery", "-A", "task_queue.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

# Scheduler stage for Celery beat
FROM base AS scheduler

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/output /app/logs /app/cache \
    && chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Scheduler command
CMD ["celery", "-A", "task_queue.celery_app", "beat", "--loglevel=info"]

# GPU-enabled stage for AI processing
FROM nvidia/cuda:12.0-devel-ubuntu22.04 AS gpu

# Configure Huawei Cloud apt mirror for faster downloads in China (Ubuntu 22.04)
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list

# Install Python and system dependencies (FFmpeg已移除，使用阿里云IMS)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    # CUDA libraries (保留用于GPU加速推理)
    libnvidia-encode-470 \
    libnvidia-decode-470 \
    # Other dependencies
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install Python packages with GPU support (Use Huawei Cloud PyPI mirror)
COPY requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple/

# Copy application
COPY . .

# Set permissions
RUN chown -R appuser:appuser /app

USER appuser

# GPU-enabled production command
CMD ["gunicorn", "app:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "600"]