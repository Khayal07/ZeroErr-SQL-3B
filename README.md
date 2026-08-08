# 🎯 ZeroErr-SQL-3B: Execution-Guided Text-to-SQL SLM

> 🚧 **Work in Progress (WIP)** — This repository is actively under development. Fine-tuning pipelines, guardrails, and benchmark workflows are being implemented.

**ZeroErr-SQL-3B** is an open-source project aimed at building a high-performance, execution-guided 3B Small Language Model (SLM) specialized in Text-to-SQL generation and error-free SQL debugging.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20QLoRA-black)
![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--3B-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 💡 Project Concept & Goals

Large foundation models often fail or introduce syntax hallucinations when generating SQL for complex database schemas. This project aims to solve that by combining a small, specialized model with an active execution guardrail.

- **Fine-Tuning**: Training `Qwen2.5-Coder-3B-Instruct` using **Unsloth (QLoRA)** on curated Text-to-SQL datasets (Spider / BIRD-SQL).
- **Self-Correction Guardrails**: Passing generated queries through a sandboxed local database. If execution errors occur, feedback is looped back into the SLM for auto-correction (up to 3 retries).
- **Local Deployment**: Quantizing the final weights to GGUF format for fast, zero-cost execution via Ollama or vLLM.
