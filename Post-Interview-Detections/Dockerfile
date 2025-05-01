# Stage 1: Builder stage with all development dependencies
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 models
RUN mkdir -p /root/.cache/torch/hub/ultralytics_yolov8_master/ && \
    python -c "from ultralytics import YOLO; YOLO('yolov8x.pt'); YOLO('yolov8n.pt')"

# Stage 2: Runtime stage with only runtime dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install only the necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set the working directory
WORKDIR /app

# Create directories for data and results
RUN mkdir -p /app/results /app/uploads

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.*/dist-packages /usr/local/lib/python3.*/dist-packages

# Copy YOLOv8 models from builder stage
COPY --from=builder /root/.cache/torch/hub/ultralytics_yolov8_master/ /root/.cache/torch/hub/ultralytics_yolov8_master/

# Copy the application code
COPY app.py .

# Verify CUDA installation
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Expose the Streamlit port
EXPOSE 8501

# Set entrypoint to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Default command
CMD ["app.py", "--server.address=0.0.0.0"]