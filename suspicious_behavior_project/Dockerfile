FROM python:3.11-slim

WORKDIR /app

# OpenCV ve PyTorch için sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<80" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python-headless || true && \
    pip install --no-cache-dir "opencv-python>=4.8.0,<5.0.0"

# Uygulama kodlarını kopyala
COPY . .

# Bulut Katmanı FastAPI ve Web Paneli portunu aç
EXPOSE 8000

ENV PYTHONPATH=/app

# Hibrit Uç/Bulut Sistemini Başlat
CMD ["python", "main.py", "--headless"]
