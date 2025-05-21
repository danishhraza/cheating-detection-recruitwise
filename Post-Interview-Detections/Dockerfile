# Stage 1: Build stage
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.2 # Check https://pytorch.org/get-started/previous-versions/ for latest CUDA 11.8 compatible
ARG TORCHVISION_VERSION=0.16.2
ARG TORCHAUDIO_VERSION=2.1.2
ARG CUDA_VERSION_TAG=cu118 # For PyTorch index URL

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUDA_VERSION_TAG

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # Set XDG_CACHE_HOME for root user as well, so pip and ultralytics cache to predictable locations
    XDG_CACHE_HOME="/root/.cache"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment (PATH is already set for venv)
RUN python${PYTHON_VERSION} -m venv /opt/venv

# Upgrade pip in virtual environment
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch, torchvision, torchaudio first for specific CUDA version
RUN pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION_TAG}

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Create cache directories first to ensure they exist, even if one remains empty
RUN mkdir -p ${XDG_CACHE_HOME}/Ultralytics ${XDG_CACHE_HOME}/torch

# Download YOLOv8 model using the installed ultralytics package
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8x.pt')"
# You can add an LS command here to debug what's in the cache if needed:
# RUN ls -R ${XDG_CACHE_HOME}

# --- Stage 2: Runtime stage ---
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION

# Create a non-root user and group first
# The -m flag creates the home directory /home/appuser
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser -s /sbin/nologin appuser

# Set environment variables for appuser
# HOME is automatically set by useradd -m and when USER is switched
# XDG_CACHE_HOME specifies where applications should store user-specific cache files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    # HOME will be /home/appuser
    XDG_CACHE_HOME="/home/appuser/.cache"

# Install essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    libgl1-mesa-glx \
    libglib2.0-0 \
    netcat-openbsd \
    curl \
    iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Create the cache directory structure for appuser before copying into it
# Ensure the parent directory $XDG_CACHE_HOME exists and has correct ownership
# useradd -m already creates /home/appuser. .cache will be a subdir.
# Docker runs COPY as root by default, so it can create subdirs in /home/appuser/.cache
# The --chown flag on COPY will then set ownership of the copied files/dirs.
# Explicitly create the target directories for clarity and to ensure structure.
USER root
RUN mkdir -p ${XDG_CACHE_HOME}/Ultralytics ${XDG_CACHE_HOME}/torch && \
    chown -R appuser:appgroup /home/appuser/.cache
# No need to switch back to USER appuser yet, COPY can run as root and --chown fixes perms.

# Copy downloaded model caches from builder
# The source directories /root/.cache/Ultralytics and /root/.cache/torch were ensured to exist in builder
COPY --from=builder --chown=appuser:appgroup /root/.cache/Ultralytics/ ${XDG_CACHE_HOME}/Ultralytics/
COPY --from=builder --chown=appuser:appgroup /root/.cache/torch/ ${XDG_CACHE_HOME}/torch/

WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup object_detect.py .
COPY --chown=appuser:appgroup gaze_detect.py .
COPY --chown=appuser:appgroup app.py .

# Ensure the app directory is owned by appuser (though COPY --chown on files might be enough)
# WORKDIR /app is created as root, then files are copied with chown.
# The directory /app itself will be owned by root.
# This explicit chown ensures /app itself is owned by appuser.
RUN chown appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]