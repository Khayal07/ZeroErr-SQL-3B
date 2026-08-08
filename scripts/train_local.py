"""Train a LoRA adapter on Qwen2.5-Coder on a local machine.

Works on:
  - NVIDIA GPU (>= 4 GB VRAM): QLoRA 4-bit via bitsandbytes (default model 1.5B).
  - CPU only: FP32 LoRA on a small dataset (slower; needs >= 12 GB RAM for 1.5B).

Usage:
    .venv/Scripts/python scripts/train_local.py \\
        --data data/chatml/train_local.jsonl \\
        --model 1.5b \\
        --out checkpoints/zeroerr-1.5b-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODELS = {
    "1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
}

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_qlora(model_id: str):
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def _load_cpu(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LoRA/QLoRA fine-tuning for Qwen2.5-Coder.")
    parser.add_argument("--data", required=True, help="ChatML jsonl dataset (field 'text')")
    parser.add_argument("--model", default="1.5b", choices=sorted(MODELS))
    parser.add_argument("--model-id", default=None, help="override with any HF model id (e.g. a tiny Qwen for smoke tests)")
    parser.add_argument("--out", default="checkpoints/zeroerr-merged")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=-1, help="train a fixed number of steps (overrides epochs); -1 = all")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    model_id = args.model_id or MODELS[args.model.lower()]

    if args.device == "cpu":
        use_cuda = False
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("requested cuda device but torch.cuda.is_available() is False")
        use_cuda = True
    else:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"[zeroerr] GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")
        try:
            model = _load_qlora(model_id)
        except ImportError as exc:
            raise SystemExit("GPU QLoRA requires bitsandbytes (pip install bitsandbytes)") from exc
    else:
        print("[zeroerr] no CUDA GPU -> training in FP32 on CPU; keep dataset small to fit RAM")
        model = _load_cpu(model_id)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        tok = tokenizer(batch["text"], truncation=True, max_length=args.max_seq)
        tok["labels"] = [list(ids) for ids in tok["input_ids"]]
        return tok

    ds = Dataset.from_json(args.data)
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir="checkpoints/zeroerr-lora",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        fp16=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"[zeroerr] merged model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())