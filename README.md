<div align="center">

# 👁️ Yapay Zeka Destekli Şüpheli Davranış ve Yüz Tanıma Sistemi
### AI-Powered Suspicious Behavior & Surveillance Analytics System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-00FFFF.svg)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20Supported-76B900.svg)](https://developer.nvidia.com/cuda-zone)

*Gerçek zamanlı video akışlarında şüpheli davranışları (oyalanma, yasaklı bölge ihlali, olağandışı hareket rotası) tespit eden ve nesne takibi yapan derin öğrenme tabanlı akıllı güvenlik analitiği sistemi.*

[Genel Bakış / Overview](#-genel-bakış--overview) • [Özellikler / Features](#-özellikler--key-features) • [Mimari / Architecture](#-sistem-mimarisi--architecture) • [Kurulum / Installation](#-kurulum-rehberi--installation) • [Yapılandırma / Configuration](#-yapılandırma--configuration) • [Kullanım / Usage](#-kullanım--usage) • [Docker](#-docker-ile-çalıştırma)

---

</div>

## 📌 Genel Bakış / Overview

**Yapay Zeka Destekli Şüpheli Davranış Tanıma Sistemi**, güvenlik kameraları (CCTV, IP kamera, RTSP akışları ve web kameraları) üzerinden insan hareketlerini analiz ederek güvenlik tehditlerini önceden belirlemek amacıyla geliştirilmiştir. 

Sistem, geleneksel hareket algılayıcıların aksine:
1. **YOLOv5** derin öğrenme mimarisi ile kişileri yüksek doğrulukla tespit eder.
2. **Centroid-Based Tracking** ve mesafe öklid algoritmalarıyla kişileri kareler boyunca kesintisiz takip eder.
3. Belirlenen kritik bölgelerde (ROI - Regions of Interest) kalış sürelerini ve yörünge verimliliğini (trajectory efficiency) hesaplayarak **oyalanma (loitering)** ve **şüpheli bekleme** davranışlarını anlık olarak raporlar.

---

## ⚡ Özellikler / Key Features

- **🎯 Gerçek Zamanlı İnsan Tespiti (YOLOv5m)**: Önceden eğitilmiş `yolov5m.pt` ağırlıkları ile yüksek FPS ve düşük gecikmeyle nesne tespiti.
- **📍 Gelişmiş Nesne Takibi (Centroid Tracker)**: Kareler arasında ID kaybetmeden kişileri benzersiz kimliklerle (`Track ID`) izleme.
- **⏱️ Oyalanma Analizi (Loitering Detection)**: Kişilerin kritik alanlarda geçirdiği süreyi hesaplama ve eşik aşıldığında alarm üretme.
- **📐 Çokgen İlgi Alanları (Polygon ROIs)**: Giriş kapıları, kasa çevreleri veya yasaklı bölgeler için serbest koordinatlı ilgi alanları tanımlama.
- **📈 Yörünge Verimlilik Analizi (Trajectory Efficiency)**: Düzensiz, amaçsız dolaşma hareketlerini doğrusal hareketten ayırt etme yeteneği.
- **🎥 Esnek Girdi Desteği**: Web kamerası (`0`), yerel video dosyaları (`.mp4`, `.avi`) veya canlı **RTSP** IP kamera akışları.
- **🐳 Docker & GPU Desteği**: NVIDIA CUDA 12.1 uyumluluğu ve Dockerfile ile anında dağıtım imkânı.

---

## 🏛️ Sistem Mimarisi / Architecture

```
                               ┌───────────────────────────┐
                               │ Video Kaynağı (RTSP/Webcam)│
                               └─────────────┬─────────────┘
                                             │ Frame Stream
                                             ▼
                               ┌───────────────────────────┐
                               │      Person Detector      │
                               │  (YOLOv5m Deep Learning)  │
                               └─────────────┬─────────────┘
                                             │ Bounding Boxes & Confidence
                                             ▼
                               ┌───────────────────────────┐
                               │       Object Tracker      │
                               │ (Centroid Distance Match) │
                               └─────────────┬─────────────┘
                                             │ Track IDs & Centroid History
                                             ▼
                               ┌───────────────────────────┐
                               │     Behavior Analyzer     │
                               │  - ROI Dwell Time         │
                               │  - Trajectory Efficiency  │
                               │  - Loitering Alarm Logic  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Canlı Görselleştirme &    │
                               │    Güvenlik Alarmları     │
                               └───────────────────────────┘
```

---

## 🛠️ Kurulum Rehberi / Installation

### Gereksinimler
- Python 3.10 veya 3.11
- NVIDIA GPU + CUDA 12.1 (Önerilir, ancak CPU üzerinde de çalışır)
- OpenCV bağımlılıkları

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/gmudz/YapayZekaDestekliSupheliTan-yanSistem.git
cd YapayZekaDestekliSupheliTan-yanSistem/suspicious_behavior_project
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin
```bash
python3 -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Yapılandırma / Configuration

Tüm sistem parametreleri `config/config.yaml` dosyasından yönetilebilir:

```yaml
# Kamera ve Akış Ayarları
camera:
  source: 0                 # 0: Webcam, "video.mp4" veya "rtsp://kullanici:sifre@ip:port/stream"
  frame:
    width: 1280
    height: 720
    fps: 30
    skip_frames: 1          # Performans için her n karede bir işle

# Model Ayarları
model:
  type: "yolov5"
  weights: "yolov5m.pt"
  confidence_threshold: 0.45
  device: "cuda:0"          # CPU için "cpu", GPU için "cuda:0"
  half_precision: true      # FP16 hızlandırma

# Takip Ayarları
tracking:
  max_age: 30               # Nesne kaybolduktan sonra silinme süresi (kare)
  min_hits: 3

# Şüpheli Davranış ve Oyalanma Ayarları
loitering_analysis:
  time_threshold_seconds: 180  # Alarm için ROI içinde geçirilen minimum saniye
  trajectory_efficiency_threshold: 0.3
  regions_of_interest:
    - name: "Ana_Giris_ROI"
      points: [[100, 150], [400, 150], [400, 450], [100, 450]]
    - name: "Yasakli_Bolge_ROI"
      points: [[500, 200], [700, 200], [700, 400], [500, 400]]
```

---

## 🚀 Kullanım / Usage

Yapılandırmanızı tamamladıktan sonra ana uygulamayı başlatın:

```bash
cd suspicious_behavior_project
python main.py
```

Uygulama çalıştığında:
- Her tespit edilen kişinin etrafında **ID etiketli sınırlayıcı kutu** belirir.
- Tanımlanan ROI bölgeleri renkli çokgenlerle ekranda çizilir.
- Şüpheli bekleme eşiği aşıldığında ekranda kırmızı uyarı uyarısı ve konsol log bildirimi oluşturulur.
- Çıkış yapmak için `q` tuşuna basın.

---

## 🐳 Docker ile Çalıştırma

Projeyi bağımlılık kurmadan Docker ile ayağa kaldırabilirsiniz:

```bash
cd suspicious_behavior_project

# Docker imajını derleyin
docker build -t suspicious-behavior-detector .

# Konteyneri başlatın (Webcam erişimi ile)
docker run --rm -it --device=/dev/video0:/dev/video0 suspicious-behavior-detector
```

*GPU desteği ile çalıştırmak için:*
```bash
docker run --gpus all --rm -it --device=/dev/video0:/dev/video0 suspicious-behavior-detector
```

---

## 📂 Dizin Yapısı / Directory Structure

```
suspicious_behavior_project/
├── config/
│   └── config.yaml          # Kamera, model, ROI ve alarm eşik parametreleri
├── modules/
│   ├── person_detector.py   # YOLOv5 nesne tespiti modülü
│   ├── object_tracker.py    # Centroid tabanlı nesne takipçisi
│   └── behavior_analyzer.py # Oyalanma ve ROI süre analiz motoru
├── utils/                   # Yardımcı çizim ve veri işleme fonksiyonları
├── yolov5m.pt               # Eğitilmiş YOLOv5m model ağırlıkları
├── Dockerfile               # Konteyner kurulum dosyası
├── requirements.txt         # Gerekli Python kütüphaneleri
└── main.py                  # Giriş noktası ve akış döngüsü
```

---

## 📄 Lisans / License

Bu proje açık kaynaklıdır ve eğitim/araştırma amaçlı geliştirilmiştir.

---

<div align="center">
  <sub>Developed by <a href="https://github.com/gmudz">Murad Ashkar (gmudz)</a></sub>
</div>
