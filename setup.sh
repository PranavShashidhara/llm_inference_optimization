#!/usr/bin/env bash
set -e

echo "Setting up Edge LLM Inference Optimization Engine..."

# ----------------------------
# 1. System dependencies
# ----------------------------
echo "Installing system dependencies..."
sudo apt update

sudo apt install -y \
    python3-venv \
    python3-pip \
    build-essential \
    git \
    cmake \
    ninja-build \
    wget

# ----------------------------
# 2. Create virtual environment
# ----------------------------
echo "Creating Python virtual environment..."

if [ -d "venv" ]; then
    echo "venv already exists, skipping creation"
else
    python3 -m venv venv
fi

# ----------------------------
# 3. Activate venv
# ----------------------------
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ----------------------------
# 4. Install Python dependencies
# ----------------------------
echo "Installing Python packages..."

pip install -r requirements.txt

# ----------------------------
# 5. Triton sanity check
# ----------------------------
echo "Checking Triton import..."
python -c "import triton; print('Triton OK')"

echo "Setup complete!"
echo "Run: source setenv.sh"