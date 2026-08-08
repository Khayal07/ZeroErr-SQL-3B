# 🎯 ZeroErr-SQL-3B: Execution-Guided Text-to-SQL SLM

> **Status: In Development** — core pipeline is scaffolded and unit-tested (37 tests, green). Fine-tuning and the end-to-end benchmark require a CUDA GPU and are the active next steps.

**ZeroErr-SQL-3B** is an open-source project aimed at building a high-performance, execution-guided 3B Small Language Model (SLM) specialized in Text-to-SQL generation and error-free SQL debugging.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20QLoRA-black)
![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--3B-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 💡 Project Concept & Goals

Large foundation models often fail or introduce syntax hallucinations when generating SQL for complex database schemas. This project solves that by combining a small, specialized model with an active execution guardrail.

- **Fine-Tuning**: Training `Qwen2.5-Coder-3B-Instruct` using **Unsloth (QLoRA)** on curated Text-to-SQL datasets (Spider / BIRD-SQL).
- **Self-Correction Guardrails**: Passing generated queries through a sandboxed local database. If execution errors occur, feedback is looped back into the SLM for auto-correction (up to 3 retries).
- **Local Deployment**: Quantizing the final weights to GGUF format for fast, zero-cost execution via Ollama.

## 🧱 Architecture

```mermaid
flowchart LR
    U[User] -->|question| API[FastAPI /v1/text2sql]
    API --> G[GuardrailLoop]
    G -->|initial SQL| E[LLM Engine]
    G -->|schema| S[SQLite Sandbox]
    G -->|execute + error| S
    S -->|ok / error hint| G
    G -->|repair prompt| E
    G -->|resolved / unresolved| API
    U -->|HTTP 200/422| API
```

1. The schema of the target database is introspected and rendered as compact DDL.
2. The SLM generates a first-pass query (ChatML prompt).
3. The query runs against a **read-only SQLite sandbox** (`mode=ro`, `query_only=ON`, watchdog timeout, SELECT/WITH allow-list, capped rows).
4. On failure the raw DB error is translated into a repair hint (nearest-column suggestions, etc.) and fed back to the model.
5. Retries up to 3 rounds; the final SQL (or an `unresolved` response) is returned.

## 🚀 Current Status (what's already built)

| Area | Status |
|------|--------|
| Data pipeline (`schema serialization`, `ChatML`, difficulty filtering, repair-pair corruption, `prep` CLI) | ✅ Done, tested |
| Sandbox guardrail (`SQLiteSandbox`, `error hints`, 3-round `GuardrailLoop`) | ✅ Done, tested |
| Inference engine (`LLMEngine` protocol + Ollama client) | ✅ Done |
| HTTP API (`/v1/text2sql`, `/health`, `/dbs`) | ✅ Done |
| Benchmark harness (`Execution Accuracy`, `VES`, base-vs-ft CLI, sqlite fixtures) | ✅ Done |
| Unit tests | ✅ 42 tests, green |
| Dataset builder (Spider + DDL schemas → ChatML, db-level train/val split) | ✅ Done, ~5.2k rows |
| Local training script (QLoRA GPU / LoRA CPU, 1.5B default) | ✅ Done |
| GGUF Q4_K_M conversion (llama.cpp + pip gguf, CPU) | ✅ Done |
| Unsloth QLoRA fine-tuning (notebook, Colab T4) | 🚧 Ready to run |
| End-to-end benchmark on Spider dev | ⏳ Pending |

## 📁 Project Structure

```
ZeroErr-SQL-3B/
├── notebooks/                  # Colab: 01 data prep · 02 Unsloth QLoRA · 03 GGUF export
├── scripts/                    # download_dataset · train_local · convert_gguf · setup_ollama
├── src/zeroerr/
│   ├── data/                   # schema serialization, ChatML, filtering, prep CLI
│   ├── guardrail/              # read-only sqlite sandbox + self-correction loop
│   ├── engine/                 # LLMEngine protocol, Ollama client
│   └── api/                    # FastAPI service
├── eval/                       # execution accuracy (EX/VES), benchmark CLI, fixtures
├── docker/                     # API + Ollama compose, Modelfile
└── tests/                      # unit tests
```

## 🖥️ Hardware Requirements

| Stage | Recommended | Notes |
|-------|-------------|-------|
| **Fine-tuning (3B)** | **Google Colab T4** | 3B QLoRA needs ≥ ~12 GB VRAM + native bf16. Not feasible on a 4 GB gaming GPU. |
| **Fine-tuning (1.5B, local)** | NVIDIA GPU ≥ 4 GB VRAM | QLoRA via `scripts/train_local.py` (~1 GB weights in 4-bit). Works on GTX 1650 Ti. |
| **Fine-tuning (1.5B, CPU)** | Any CPU, ≥ 12 GB RAM | Much slower FP32 LoRA; only worth it for small datasets. |
| **Inference** | Any machine with Ollama (CPU ok) | Q4_K_M GGUF runs comfortably on a 4 GB GPU / CPU. |
| **Guardrail + API + eval** | Any machine | CPU-only, 42 unit tests run without GPU/network. |

## 💻 Everything runs on your machine (no Colab)

For a 4 GB NVIDIA GPU the trained model is the **1.5B** variant. Everything below is
fully local and CPU-capable at every step.

```bash
# 1) one-time env setup (CUDA-enabled torch for your GPU)
make install-local

# 2) fetch Spider + DDL schemas, build a compact ChatML dataset
make data
make prep-small          # ~1.8k balanced rows + repair pairs

# 3) train a LoRA/QLoRA adapter and merge it (1.5B default, ~30-90 min on 1650 Ti)
make train-local

# 4) convert merged weights to GGUF Q4_K_M (CPU) and register with Ollama
make gguf
make ollama-import

# 5) verify with the guardrail + benchmark
.\.venv\Scripts\python -m uvicorn zeroerr.api.main:app --port 8000
.\.venv\Scripts\python eval/run_benchmark.py --smoke
```

Expected times (Ryzen 5 / 1650 Ti 4 GB): 1.5B QLoRA at `max_seq=512`,
`batch 1 × grad-accum 8` — roughly 30–80 min for 1–2 epochs. CPU-only is
several hours; keep `--per-bucket` small.

## 🧪 Quickstart

```bash
pip install -e ".[dev]"

# run the unit tests (no GPU / no network needed)
pytest

# build the fixture databases and smoke-benchmark the guardrail
python eval/run_benchmark.py --smoke
```

Start the API against a local Ollama model:

```bash
uvicorn zeroerr.api.main:app --host 0.0.0.0 --port 8000 --reload
curl -X POST http://localhost:8000/v1/text2sql \
  -H "Content-Type: application/json" \
  -d '{"question": "list departments", "database_id": "department_emp"}'
```

## 🛠️ Next Steps (roadmap)

1. `scripts/download_data.sh` to fetch Spider (+ BIRD dev), then `python -m zeroerr.data.prep` to emit the ChatML dataset.
2. Train QLoRA in `notebooks/02_unsloth_qlora.ipynb` (CUDA GPU, WSL2 recommended on Windows).
3. Export Q4_K_M GGUF in `notebooks/03_gguf_export.ipynb` and register via `scripts/setup_ollama.sh`.
4. Measure base vs fine-tuned accuracy with `eval/run_benchmark.py` and publish the metrics here.

## 📜 License

MIT — see [LICENSE](LICENSE).
