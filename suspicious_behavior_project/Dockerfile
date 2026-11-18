FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV & PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<80" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python-headless || true && \
    pip install --no-cache-dir "opencv-python>=4.8.0,<5.0.0"

# Copy application code
COPY . .

# Expose Cloud Layer FastAPI & Web Dashboard port
EXPOSE 8000

ENV PYTHONPATH=/app

# Launch Hybrid Edge/Cloud System
CMD ["python", "main.py", "--headless"]
