@echo off
REM ==============================================================================
REM TUBITAK 2209-A: Yapay Zeka Destekli Hibrit Mimarili Supheli Davranis Tespiti
REM ve Adil Yuz Tanima Sistemi — Windows Kurulum ve Calistirma Betigi (run.bat)
REM ==============================================================================

cd /d "%~dp0\suspicious_behavior_project"

echo ==============================================================================
echo  TUBITAK 2209-A Hibrit Supheli Davranis ve Adil Yuz Tanima Sistemi
echo ==============================================================================

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Sanal ortam olusturuluyor...
    python -m venv venv
)

echo [2/3] Kutuphaneler kontrol ediliyor...
venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<80"
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip uninstall -y opencv-python-headless 2>nul
venv\Scripts\python.exe -m pip install "opencv-python>=4.8.0,<5.0.0"

echo [3/3] Sistem baslatiliyor (Web Paneli: http://localhost:8000)...
start "" http://localhost:8000
venv\Scripts\python.exe main.py %*
pause
