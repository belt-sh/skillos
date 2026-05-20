# HRM-Text-1B SkillOS pilot (experiment)

Parallel experiment: can we run SkillOS GRPO on `sapientinc/HRM-Text-1B`
instead of Qwen3-8B?

## Why bother

- 1B params vs 8B → fits 4–8× in a 96 GB GPU, makes parallel ablations realistic
- PrefixLM pretraining objective (instruction bidirectional, response causal)
  is architecturally a *better* fit for "given trajectory + skill library,
  emit tool calls + skill markdown" than Qwen3's pure causal chat.
- HRM-Text-1B's reasoning benchmark numbers (GSM8K 84.5, MATH 56.2) are
  strong for a 1B — competitive with Qwen3.5 2B at much smaller scale.
- Released last week (May 2026), fully open: weights + code + data recipe.

## Why it's risky

| risk | mitigation |
|---|---|
| Pre-alignment checkpoint — **not** instruction-tuned. Won't emit `<tool_call>` blocks out of the box. | SFT warmup phase before GRPO. ~50–200 (trajectory → tool-call + skill) pairs replayed from `output/skillos-live/rollouts.jsonl`. |
| PrefixLM mask needs `token_type_ids` to match training distribution. | Adapter in `train_hrm.py` adds `token_type_ids = ones_like(input_ids)` for the prompt block. |
| Custom model class — not native in transformers 5.8.1. | `trust_remote_code=True` picks up the bundled `modeling_hrm_text.py`. Verified the file is in the checkpoint. |
| Condition prefix tokens (`direct`, `cot`, `noisy`, `synth`) are required for good outputs. | Use `synth,cot` composite condition for the curator prompt (reasoning + structured output style). |
| PEFT / LoRA: `target_modules="all-linear"` may not find HRM's gated attention modules cleanly. | First smoke test will print all `nn.Linear` modules to verify PEFT can attach. Fallback: explicit module list. |
| Comparison-to-paper breaks (paper used Qwen3-8B). | This is an *exploration*, not a reproduction. Report as separate result. |

## Plan

1. ✅ Download `sapientinc/HRM-Text-1B` to `HRM-Text-1B/`
2. ✅ Clone reference repo to `HRM-Text-repo/` (FSDP2 trainer, eval harness)
3. **GPU is free → 5-min smoke test**: load model, generate 1 completion with
   `token_type_ids`, verify it produces text.
4. **PEFT compatibility check**: print all `nn.Linear` layer names; verify
   `target_modules="all-linear"` covers attention + MLP.
5. **SFT warmup**: replay ~50 (curator_input → tool_calls) pairs from existing
   `output/skillos-live/rollouts.jsonl`. ~5 minutes on this GPU.
6. **GRPO pilot**: same SkillOS recipe (executor + judge via inference.sh,
   composite reward, 8 generations per prompt), but on HRM-Text-1B + LoRA.

## Filesystem layout

```
experiments/hrm_pilot/
├── HRM-Text-1B/              # weights + custom model code (2.4 GB)
├── HRM-Text-repo/            # reference trainer + eval (FSDP2, FA3)
├── configs/
│   └── hrm_pilot.yaml        # SkillOS config adapted for HRM
├── README.md                 # this file
├── download.log
└── clone.log
```

## Status

- model: downloaded (2.4 GB)
- code: cloned
- venv: shares main `.venv` (transformers 5.8.1 — bundled remote code provides hrm_text)
- smoke test: pending (waiting for current LoRA pilot to free GPU headroom)
- SFT warmup: pending
- GRPO pilot: pending
