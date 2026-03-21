#!/bin/bash
# ============================================================
# TTS-specific Docker setup for Wall-E (KhanomTan TTS)
# Installs only what the TTS container needs.
# Run inside Docker image build (no sudo needed — runs as root).
# ============================================================
set -e

echo "========================================"
echo "   Wall-E TTS Setup (KhanomTan TTS)"
echo "========================================"

# ── 1. System dependencies ───────────────────────────────────
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Upgrade pip/setuptools/wheel first ────────────────────
echo ""
echo "[2/6] Upgrading pip, setuptools, and wheel..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 3. Pre-install pandas 1.5.3 ──────────────────────────────
# pandas 1.4.x has NO binary wheel for Python 3.11 — pip would try to build
# it from source, which fails because the isolated build env lacks pkg_resources.
# pandas 1.5.3 (last 1.x release) DOES have a cp311 binary wheel AND satisfies
# Coqui TTS's constraint (pandas>=1.1.0,<2.0), so we pin it here first.
echo ""
echo "[3/6] Pre-installing pandas 1.5.3 (Python 3.11 binary wheel)..."
pip install --no-cache-dir 'pandas==1.5.3'

# ── 4. PyTorch (CUDA 12.1 build) ─────────────────────────────
echo ""
echo "[4/6] Installing PyTorch (CUDA 12.1)..."
pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 5. Python packages (unified requirements.txt) ────────────
echo ""
echo "[5/6] Installing Python packages from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# ── 6. KhanomTan TTS (PyThaiTTS + Coqui TTS) ─────────────────
# Installed last so pandas is already satisfied and no source-build is triggered.
echo ""
echo "[6/6] Installing KhanomTan TTS dependencies (pythaitts + TTS)..."
pip install --no-cache-dir pythaitts TTS

echo ""
echo "========================================"
echo "   TTS Setup complete!"
echo "========================================"
