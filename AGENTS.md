# ZeroErr-SQL-3B — Project Memory (opencode)

Goal: execution-guided Text-to-SQL SLM; base Qwen2.5-Coder-1.5B/3B, QLoRA fine-tune,
Ollama GGUF, guardrail API + eval.

## Current status (as of 2026-08-08)
- 42 unit tests green; no GPU/network needed.
- **Everything validated except the full training run** — that is the only remaining step.
- Local GPU QLoRA verified end-to-end on the user's GTX 1650 Ti (4 GB):
  4-bit 1.5B load, ~10 s/it at `max_seq=384`, LoRA merge OK, no OOM.
- GGUF pipeline verified on CPU: `q8_0` default → valid GGUF (Q8_0 tensors).
- Data ready: `data/chatml/train_local.jsonl` (~1895 rows, leakage-free splits).
- Python venv at `.venv` has torch 2.6.0+cu124, bitsandbytes, peft, gguf.

## Remaining step (user runs it when ready — do NOT auto-launch a 5h job)
```bash
make train-local        # ~5h on this machine → checkpoints/zeroerr-1.5b-merged
make gguf               # q8_0 GGUF
make ollama-import      # must run inside WSL (Ollama is a Linux binary)
```
Before launching a long run, CONFIRM with the user — earlier a 5h run was started and user
stopped it ("cox gec oldu"). Prefer a background process + log file + progress check.

## Hard-won gotchas (do not repeat)
1. `make install-local`: pip must use `--force-reinstall --no-deps torch --index-url
   https://download.pytorch.org/whl/cu124` — a plain `pip install torch` is a NO-OP because
   CPU torch already satisfies the requirement, silently leaving a CPU build
   (`torch.cuda.is_available() == False`). CUDA 12.9 driver is present, cu124 wheels work.
2. `scripts/convert_gguf.py`: the llama.cpp python converter only supports
   f32/f16/bf16/q8_0/tq1_0/tq2_0. k-quants like `q4_K_M` need the C++ `llama-quantize` on PATH;
   the script refuses and suggests `q8_0`. Keep q8_0 as the local default.
3. Ollama import must run under WSL (`bash scripts/setup_ollama.sh`); Windows has no ollama CLI.
4. Training defaults: `--epochs 1 --max-seq 512 --batch-size 1 --grad-accum 8` ⇒
   ~1895/8 ≈ 237 optimizer steps; at seq 384 that's ~5h. Long foreground runs block the session —
   launch with `Start-Process -RedirectStandardOutput logs\...` (logs are buffered, empty for a while).
5. Smoke-test artifacts follow `_smoke` naming; clean up `data/chatml/_smoke.jsonl`,
   `checkpoints/zeroerr-*` after verifying.
6. `third_party/` = shallow llama.cpp clone; gitignored, re-created by the converter if missing.
   Pip `-e` install of `gguf-py` happens on demand.
7. HF Hub unauthenticated warning ("set HF_TOKEN") is harmless for these public models.

## Repo layout (paths used by tools)
- `scripts/`: train_local.py (QLoRA/LoRA, `--device auto|cuda|cpu`, `--model-id` for tiny smoke),
  convert_gguf.py, download_dataset.py, setup_ollama.sh (WSL), prep via `python -m zeroerr.data.prep`.
- `eval/`: run_benchmark.py (--smoke uses sqlite fixtures, no network), fixtures.build.
- `zeroerr/api/main.py`: FastAPI guardrail; `tests/` are the 42 green tests.
- Makefile targets: `data prep-small train-local gguf ollama-import fixtures test lint bench api`.
