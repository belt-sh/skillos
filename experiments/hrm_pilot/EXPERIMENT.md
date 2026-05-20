# Experiment: HRM-Text-1B as the SkillOS curator

**Status:** scaffold complete, GPU smoke test pending
**Date opened:** 2026-05-20
**Author:** okaris

---

## Hypothesis

A 1B-parameter PrefixLM-pretrained model (`sapientinc/HRM-Text-1B`) can serve
as the SkillOS skill curator instead of Qwen3-8B, with three potentially
significant advantages:

1. **Architectural fit.** HRM-Text was pre-trained on instruction–response
   pairs with a PrefixLM mask: the instruction tokens attend bidirectionally
   while the response is autoregressive. The SkillOS curator's job has
   exactly this shape — *given* (trajectory + skill library + system prompt),
   *emit* (tool call + skill markdown). Qwen3-8B is causal-only and was
   pre-trained on raw text, so the prompt → response shape is a less direct
   match to its training distribution.

2. **8× smaller, 8× more iterable.** At 1B parameters, HRM-Text fits 4–6× on
   a 96 GB GPU even with bf16 + LoRA + ref model + KV cache. This makes
   parallel ablation sweeps (lr, beta, r/alpha, condition prefix) actually
   feasible on our single Blackwell box, vs. one-at-a-time with Qwen3-8B.

3. **Strong reasoning at the 1B scale.** HRM-Text-1B reports GSM8K 84.5,
   MATH 56.2, ARC-C 81.9, MMLU 60.7 — competitive with Qwen3.5 2B and Llama3.2
   3B. The recurrent dual-timescale (H/L) architecture appears to give it
   more "effective depth" per parameter, which is plausibly useful for a
   curator that has to reason over a multi-step trajectory and decide what
   skill to extract.

---

## Why this is *not* just "use a smaller model"

The point isn't size, it's architectural match. Qwen3.5 2B and Llama3.2 3B
are also small, but they're causal-only and don't share HRM-Text's
prefix-bidirectional pre-training. The SkillOS curator is conceptually an
encoder–decoder task (read context, write skill), and HRM-Text is the only
small open model we've seen that bakes that shape into pretraining.

If HRM-Text-1B reaches similar SkillOS reward as Qwen3-8B at much lower
wall-clock and FLOPs, it would be evidence that **task-architecture fit
matters more than scale** for this particular RL setup.

---

## Method

### Phase 1 — SFT warmup (~5 min on this GPU)

HRM-Text-1B is explicitly *pre-alignment* — it is **not** instruction-tuned
and has never emitted a `<tool_call>...</tool_call>` block. GRPO has no
gradient signal to bootstrap from if the base policy can't produce any
parseable tool calls at all.

We bootstrap with a small supervised pass:
- Replay ~50–200 successful curator outputs from a prior Qwen3-8B SkillOS
  run's `output/skillos-live/rollouts.jsonl`.
- Format each as `(curator_input, tool_call_emission)` pair.
- Wrap the curator_input in `<|im_start|>{synth,cot prefix}…<|im_end|>` and
  pass `token_type_ids=1` over the whole prompt block (PrefixLM mask).
- Standard SFT NLL loss on the response portion only.
- LoRA r=32 alpha=64 (same as Phase 2).

Goal: get the model emitting *syntactically valid* tool calls before GRPO
starts. Not asking it to be good — just functional enough for the GRPO
reward to discriminate samples.

### Phase 2 — GRPO with the SkillOS recipe

Same training loop as `configs/alfworld_lora_pilot.yaml` from the main
project, with these substitutions:

| knob | Qwen3-8B pilot | HRM-Text-1B pilot |
|---|---|---|
| base model | `Qwen/Qwen3-8B` | `sapientinc/HRM-Text-1B` (local) |
| `trust_remote_code` | n/a | `true` |
| condition prefix | n/a | `synth,cot` (composite reasoning prefix) |
| PrefixLM mask | n/a | `token_type_ids = 1` on prompt block |
| `learning_rate` | 1e-6 (paper) | **5e-5** (50× — LoRA on a 1B fresh-pre-trained model needs faster movement) |
| `beta` (KL coef) | 0.001 (was wrong) | **0** (paper: "we discard the KL term") |
| `max_completion_length` | 4096 (paper) | 4096 (HRM trained at 4096, don't extrapolate) |
| `num_generations` | 8 | 8 |
| `batch_size × grad_accum` | 2 × 16 = 32 | 2 × 16 = 32 |

Executor (Qwen3-8B) and judge (Qwen3-32B) stay on inference.sh — cloud
already optimizes those. Only the curator runs locally.

---

## Risks (and how we mitigate)

| risk | likelihood | mitigation |
|---|---|---|
| HRM model class not in transformers 5.8.1 (no native registration) | confirmed | `trust_remote_code=True` picks up the bundled `modeling_hrm_text.py`. No transformers upgrade needed. |
| `target_modules="all-linear"` misses HRM's gated-attention linear projections | unknown | `smoke_test.py` lists every `nn.Linear` submodule and prints them — we'll see exactly what PEFT can attach to before committing to a run. |
| Missing `token_type_ids` silently falls back to pure causal — model produces worse logits | known | Adapter in train script always passes `token_type_ids=ones_like(input_ids)` for the prompt block. |
| Condition prefix tokens are wrong type — model treats `<\|quad_end\|>` etc. as garbage | known | We use the README's published mapping: `synth` → `<\|quad_end\|>`, `cot` → `<\|object_ref_end\|>`, composed in order as a single bidirectional prefix. |
| Phase 1 SFT bakes in bad habits (memorizes specific skills instead of learning the format) | medium | Use small N (~50–200), low epoch count (1–2), small LoRA — explicitly aiming for *format induction*, not behavior cloning. |
| GRPO reward signal is too noisy at 1B to teach skill curation | unknown | This is the experiment. If reward doesn't trend up over 10 opt steps the way it does for Qwen3-8B, we've answered the question — negatively. |
| Comparison to paper Table 1 is broken (paper used Qwen3-8B) | by design | Report as a *separate result*, not as a SkillOS reproduction. Goal is the curve shape, not the absolute number. |

---

## Success criteria

This is an exploration, not a benchmark race. We declare it *worth pursuing*
if any of the following hold after a 10-step LoRA pilot:

1. **Reward curve trends up similarly** to Qwen3-8B LoRA pilot — within 0.1
   of its final mean reward after the same number of optimizer steps.
2. **Wall-clock is ≤ 1/3** of Qwen3-8B at matched effective batch.
3. **Tools/failure_frequency stays near 0** after Phase 1 SFT — proving the
   format induction worked.

If none of these hold, we stop and write up why. If all three hold, we scale
to a longer run.

---

## Filesystem layout (this folder)

```
experiments/hrm_pilot/
├── HRM-Text-1B/              # 2.4 GB model weights (gitignored, reproducible via HF Hub)
├── HRM-Text-repo/            # cloned upstream reference (gitignored)
├── configs/
│   └── hrm_pilot.yaml        # Phase 2 GRPO config
├── README.md                 # quick status / orientation
├── EXPERIMENT.md             # this document — full design + rationale
├── smoke_test.py             # 5-min GPU check: load, generate, list nn.Linear
├── download.log              # gitignored
└── clone.log                 # gitignored
```

---

## Open questions (revisit after smoke test)

- Does `model.generate()` work at all on the bundled remote code with our
  transformers 5.8.1? (Risk: the bundled code may target a different
  transformers API minor version.)
- What VRAM does HRM-Text-1B + LoRA r=32 + ref model + 256-completion KV
  cache actually use? (Estimate: ~10–14 GB. If real, we can run 4–5 in
  parallel.)
- Does `token_type_ids` propagate through `model.generate()` correctly, or
  does TRL drop it during sampling?
- What does the warmup-SFT data look like? Are 50 examples enough? Should we
  include negative examples (malformed tool calls) so the model learns the
  boundary?

These get answered in the smoke test + Phase 1 SFT run.
