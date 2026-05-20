"""Sanity check: download + load Qwen3-8B, wrap with LoRA, do one forward pass.

No API spend, no GRPO loop — just confirms VRAM math holds on the 96GB GPU
before we commit to the multi-day paper run.
"""

from __future__ import annotations

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_LEN = 4096


def mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def main():
    assert torch.cuda.is_available(), "Need CUDA"
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {name} (total {mb(total)})")

    print(f"\nLoading tokenizer {MODEL}…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  tokenizer ready in {time.time() - t0:.1f}s")

    print(f"\nLoading model {MODEL} (bf16, device_map=cuda)…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    print(f"  model ready in {time.time() - t0:.1f}s")
    print(f"  VRAM after model load: {mb(torch.cuda.memory_allocated())}")

    print("\nApplying LoRA r=32 alpha=64 (all-linear, causal LM)…")
    peft_cfg = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")
    print(f"  VRAM after LoRA wrap: {mb(torch.cuda.memory_allocated())}")

    print("\nEnabling gradient checkpointing…")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(f"\nForward + backward with seq_len={MAX_LEN}…")
    text = "Solve the following ALFWorld task. " * 50
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding="max_length").to(device)
    enc["labels"] = enc["input_ids"].clone()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(**enc)
    out.loss.backward()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    print(f"  fwd+bwd in {dt:.2f}s, peak VRAM: {mb(peak)} / {mb(total)} ({100 * peak / total:.1f}%)")
    print(f"  loss = {out.loss.item():.4f}")

    print("\nOK — Qwen3-8B + LoRA + grad-checkpointing fits.")


if __name__ == "__main__":
    main()
