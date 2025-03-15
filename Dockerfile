FROM python:3.9-slim

# Install system dependencies required for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and extract the shape predictor file
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Expose port for the Flask app
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]