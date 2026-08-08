"""Convert a merged HuggingFace model to GGUF for Ollama, on CPU.

Uses llama.cpp's ``convert_hf_to_gguf.py`` + the pip ``gguf`` package, so no
C/C++ toolchain is required.

Quantization:
  - ``q8_0`` / ``f16`` / ``bf16``   done directly by the llama.cpp converter (pure Python).
  - k-quants like ``q4_K_M`` require llama.cpp's C++ ``llama-quantize`` on PATH ;
    if it is not found this script will refuse and suggest ``q8_0`` instead.

Usage:
    .venv/Scripts/python scripts/convert_gguf.py \\
        --input checkpoints/zeroerr-1.5b-merged \\
        --output gguf/zeroerr-1.5b-q8_0.gguf \\
        --ftype q8_0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_DIR = Path("third_party") / "llama.cpp"
LLAMA_REPO = "https://github.com/ggerganov/llama.cpp.git"

DIRECT_TYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}


def _ensure_llama_cpp() -> Path:
    script = LLAMA_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        print(f"[zeroerr] cloning llama.cpp -> {LLAMA_DIR}")
        LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR)], check=True)
    gguf_py = LLAMA_DIR / "gguf-py"
    if (gguf_py / "pyproject.toml").exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(gguf_py)],
            check=True,
        )
    return script


def _quantize_cpp(input_gguf: Path, output_gguf: Path, ftype: str) -> bool:
    binary = shutil.which("llama-quantize")
    if binary is None:
        return False
    subprocess.run([binary, str(input_gguf), str(output_gguf), ftype], check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a merged HF model to GGUF.")
    parser.add_argument("--input", required=True, help="merged HuggingFace model directory")
    parser.add_argument("--output", default="gguf/zeroerr.q8_0.gguf")
    parser.add_argument("--ftype", default="q8_0", help=f"direct: {sorted(DIRECT_TYPES)}; k-quants: q4_K_M (needs llama-quantize)")
    args = parser.parse_args()

    script = _ensure_llama_cpp()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ftype = args.ftype.lower()

    if ftype in DIRECT_TYPES:
        subprocess.run([sys.executable, str(script), str(Path(args.input)), "--outfile", str(out), "--outtype", ftype], check=True)
    else:
        tmp16 = out.with_name(out.stem + ".f16.gguf")
        subprocess.run([sys.executable, str(script), str(Path(args.input)), "--outfile", str(tmp16), "--outtype", "f16"], check=True)
        if not _quantize_cpp(tmp16, out, ftype):
            tmp16.unlink()
            raise SystemExit(
                f"'llama-quantize' not found on PATH to produce {ftype}. "
                "Use --ftype q8_0 (fully local) or build llama.cpp and re-run."
            )
        tmp16.unlink()

    print(f"[zeroerr] GGUF written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())