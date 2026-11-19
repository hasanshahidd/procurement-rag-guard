#!/usr/bin/env sh
# One-command local setup on Linux or macOS: virtual environment, packages, local models, tests.
#   sh scripts/setup.sh            # API and tests
#   WITH_ML=1 sh scripts/setup.sh  # also torch, transformers, peft
set -e
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements-dev.txt
if [ "${WITH_ML:-0}" = "1" ]; then
  "$VENV/bin/python" -m pip install --quiet -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install it from https://ollama.com/download, then rerun."
  exit 1
fi
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

"$VENV/bin/python" -m pytest -q
echo
echo "Ready. Start the API:  $VENV/bin/python -m uvicorn app.main:app --port 8000"
echo "Then open http://127.0.0.1:8000/ (web UI) or http://127.0.0.1:8000/docs (OpenAPI)."
