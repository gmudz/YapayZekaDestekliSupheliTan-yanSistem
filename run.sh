#!/usr/bin/env bash
# ==============================================================================
# TÜBİTAK 2209-A: Yapay Zeka Destekli Hibrit Mimarili Şüpheli Davranış Tespiti
# ve Adil Yüz Tanıma Sistemi — Tek Komutla Kurulum ve Çalıştırma Betiği (run.sh)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/suspicious_behavior_project"
cd "${PROJECT_DIR}"

echo "=============================================================================="
echo " 🛡️ TÜBİTAK 2209-A Hibrit Şüpheli Davranış ve Adil Yüz Tanıma Sistemi"
echo "=============================================================================="

# 1. Ensure uv or Python 3.11+ venv is available
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/4] Hızlı Python paket yöneticisi (uv) kuruluyor..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x "./venv/bin/python" ]; then
    echo "[2/4] Python 3.11 sanal ortamı (venv) oluşturuluyor..."
    uv venv venv --python 3.11 --seed
fi

# 2. Install dependencies if not yet installed
if ! ./venv/bin/python -c "import torch, cv2, fastapi, pymilvus, yolov5" >/dev/null 2>&1; then
    echo "[3/4] Gerekli yapay zeka ve bulut kütüphaneleri yükleniyor..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        uv pip install --python ./venv/bin/python torch torchvision
    else
        uv pip install --python ./venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    uv pip install --python ./venv/bin/python -r requirements.txt
    uv pip uninstall --python ./venv/bin/python opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true
    uv pip install --python ./venv/bin/python "setuptools>=68.0.0,<80.0.0" "opencv-python>=4.8.0,<5.0.0"
else
    echo "[3/4] Bağımlılıklar hazır ✅"
fi

# 3. Start Milvus Standalone container if Docker is running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "[Opsiyonel] Docker tespit edildi — Milvus Standalone kontrol ediliyor..."
    docker compose up -d 2>/dev/null || true
fi

echo "[4/4] Sistem başlatılıyor (Web Paneli: http://localhost:8000)..."
exec ./venv/bin/python main.py "$@"
