#!/bin/bash
# ============================================================
# Docker-friendly setup script for Wall-E
# Replaces manual requirements-*.txt files.
# Run inside Docker image build (no sudo needed — runs as root).
# ============================================================
set -e

echo "========================================"
echo "   Wall-E Docker Setup"
echo "========================================"

# ── 1. System dependencies ───────────────────────────────────
echo ""
echo "[1/4] Installing system dependencies..."
apt-get update && apt-get install -y \
    ffmpeg \
    zstd \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    ca-certificates \
    curl \
    git \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# ── 2. PyTorch (CUDA 12.1 build) ─────────────────────────────
echo ""
echo "[2/4] Installing PyTorch (CUDA 12.1)..."
pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. NeMo ASR (Typhoon STT) ────────────────────────────────
echo ""
echo "[3/4] Installing NeMo toolkit for Typhoon ASR..."
pip install --no-cache-dir "nemo-toolkit[asr]==2.4.0"

# ── 4. Python packages (unified requirements.txt) ────────────
echo ""
echo "[4/4] Installing Python packages from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# ── 5. VachanaTTS (Thai VITS TTS) ────────────────────────────
echo ""
echo "[5/5] Installing VachanaTTS..."
VACHANA_DIR="/app/VachanaTTS"
if [ ! -d "$VACHANA_DIR" ]; then
    git clone https://github.com/VYNCX/VachanaTTS.git "$VACHANA_DIR"
fi
cd "$VACHANA_DIR"
if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
fi
# Make VachanaTTS importable as a package
SITE_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE_DIR/vachanatts"
rsync -a --exclude='.git' . "$SITE_DIR/vachanatts/"
cd /app
echo "VachanaTTS installed."

echo ""
echo "========================================"
echo "   Setup complete!"
echo "========================================"
