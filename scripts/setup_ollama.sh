#!/usr/bin/env bash
# Register the exported GGUF with Ollama (see notebooks/03_gguf_export.ipynb).
set -euo pipefail

GGUF="${1:-gguf/zeroerr-3b.unsloth.Q4_K_M.gguf}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama is not installed or not on PATH"
    exit 1
fi
if [[ ! -f "$GGUF" ]]; then
    echo "GGUF not found: $GGUF (run the export notebook first)"
    exit 1
fi

TAG="zeroerr:3b"
sed "s|FROM .*|FROM $GGUF|" docker/Modelfile.zeroerr | ollama create "$TAG"

echo "pulled $TAG"
ollama list | grep "$TAG"