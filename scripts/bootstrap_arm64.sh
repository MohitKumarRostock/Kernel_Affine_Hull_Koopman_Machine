#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv-arm64}"
PYTHON_BIN="/opt/homebrew/opt/python@3.14/bin/python3.14"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Kernel Affine Hull Koopman Machine"
echo "ARM64 reference-environment bootstrap"
echo

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: This bootstrap targets Apple Silicon / arm64."
    echo "Detected architecture: $(uname -m)"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Native Python 3.14 was not found at:"
    echo "$PYTHON_BIN"
    echo
    echo "Install it with:"
    echo "  /opt/homebrew/bin/brew install python@3.14"
    exit 1
fi

PY_ARCH="$("$PYTHON_BIN" -c 'import platform; print(platform.machine())')"

if [[ "$PY_ARCH" != "arm64" ]]; then
    echo "ERROR: Python is not native arm64."
    echo "Detected Python architecture: $PY_ARCH"
    exit 1
fi

if [[ -e "$ENV_DIR" ]]; then
    echo "ERROR: Target environment already exists:"
    echo "$ENV_DIR"
    echo "Nothing was deleted."
    exit 1
fi

echo "Python:"
"$PYTHON_BIN" --version
echo "Architecture: $PY_ARCH"
echo "Environment: $ENV_DIR"
echo

"$PYTHON_BIN" -m venv "$ENV_DIR"

"$ENV_DIR/bin/python" -m pip install --requirement requirements-lock-arm64.txt

echo
echo "Environment created successfully."
echo "Activate with:"
echo "  source $ENV_DIR/bin/activate"
