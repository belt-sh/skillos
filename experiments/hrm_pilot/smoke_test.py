"""HRM-Text-1B smoke test.

Run when the main GPU has at least ~6 GB free.

Checks:
1. Model loads via trust_remote_code (no transformers source patching needed).
2. PrefixLM token_type_ids forward path works.
3. .generate() produces non-empty output under the `synth,cot` condition.
4. Lists every nn.Linear submodule by name so we can confirm whether PEFT's
   `target_modules="all-linear"` will find the right ones — and what to put
   in an explicit list if it doesn't.
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "/home/ok/skillos/experiments/hrm_pilot/HRM-Text-1B"


def main() -> None:
    print(f"loading {MODEL_DIR} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e9:.2f}B")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # synth,cot condition (reasoning + structured-output style)
    condition = "<|quad_end|><|object_ref_end|>"
    prompt = (
        f"<|im_start|>{condition}"
        "You are a skill curator. Given an agent trajectory, write a short "
        "markdown skill with YAML frontmatter that captures one reusable "
        "insight. Output ONLY the markdown.<|im_end|>"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

    print(f"\nprompt tokens: {inputs['input_ids'].shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    print(
        f"\ngenerated {new_tokens} tokens in {gen_time:.1f}s "
        f"({new_tokens/gen_time:.1f} tok/s)"
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    print("\n--- generated output ---")
    print(decoded)
    print("--- end ---\n")

    # PEFT target_modules audit
    print("=== nn.Linear submodules (for PEFT target_modules) ===")
    linear_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            # Take the leaf attribute name (last segment) — that's what PEFT
            # matches against when you pass a list of strings.
            leaf = name.rsplit(".", 1)[-1]
            linear_names.add(leaf)
    for n in sorted(linear_names):
        print(f"  {n}")

    print(
        "\nIf the list above contains the usual attention/MLP names "
        "(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj or "
        "their HRM equivalents), `target_modules=\"all-linear\"` should work."
    )


if __name__ == "__main__":
    sys.exit(main())
