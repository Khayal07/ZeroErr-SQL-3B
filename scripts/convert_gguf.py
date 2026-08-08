"""Convert a merged HuggingFace model to a GGUF (Q4_K_M) for Ollama, on CPU.

Uses llama.cpp's ``convert_hf_to_gguf.py`` + the pip ``gguf`` package, so no
C/C++ toolchain is required. The llama.cpp repo is cloned once into
``third_party/``.

Usage:
    .venv/Scripts/python scripts/convert_gguf.py \\
        --input checkpoints/zeroerr-1.5b-merged \\
        --output gguf/zeroerr-1.5b-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LLAMA_DIR = Path("third_party") / "llama.cpp"
LLAMA_REPO = "https://github.com/ggerganov/llama.cpp.git"


def _ensure_llama_cpp() -> Path:
    script = LLAMA_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        print(f"[zeroerr] cloning llama.cpp -> {LLAMA_DIR}")
        LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR)], check=True)
    gguf_py = LLAMA_DIR / "gguf-py"
    if (gguf_py / "setup.py").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(gguf_py)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "gguf"], check=True)
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a merged HF model to GGUF Q4_K_M.")
    parser.add_argument("--input", required=True, help="merged HuggingFace model directory")
    parser.add_argument("--output", default="gguf/zeroerr.q4_k_m.gguf")
    parser.add_argument("--ftype", default="q4_K_M", help="gguf ftype (e.g. q4_K_M, q8_0, f16)")
    args = parser.parse_args()

    script = _ensure_llama_cpp()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(script),
            str(Path(args.input)),
            "--outfile",
            str(out),
            "--outtype",
            args.ftype,
        ],
        check=True,
    )
    print(f"[zeroerr] GGUF written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())