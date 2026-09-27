#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
# CPU wheel: PyBullet training does not use GPU, and default torch pulls CUDA (~2GB).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
# onnx pulls numpy>=2.0 as a transitive dep; restore the pin gym-pybullet-drones needs.
python -m pip install "numpy<2.0"
echo "venv ready. Activate with: source train/.venv/bin/activate"
