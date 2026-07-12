# Contributing

This is an independent reproduction repo. The primary artifacts are the reproducibility report (`docs/repro_report.md`), the running JOURNAL (`JOURNAL.md`), and the full deviation ledger (`DIVERGENCES.md`). The workflow below is what actually gets used in practice.

## before opening a PR

- Reproduce whatever result you're touching. For any change that affects held-out numbers, re-run at least the no-memory arm on ALFWorld and confirm the paired-McNemar comparator produces the expected pattern.
- Update `DIVERGENCES.md` if the change alters the alignment with the paper (either fixes a deviation or introduces one).
- Update `JOURNAL.md` with the observed result if the change closes or opens a claim. Aggregate numbers only for GPQA-Diamond (see the `gpqa-no-public-leak` note).

## running the eval sweeps

Every held-out sweep uses the same pattern:

1. Confirm the executor endpoint is healthy with a concurrency-matched probe (single + 10-burst + 40-burst). Storm-resilient supervisors under `scripts/*_sweep_supervisor.sh` do this automatically.
2. Pair every arm against the **canonical** no-memory baseline (`output/eval-pathbv4/no_memory.jsonl`, 33.6% SR). Do not reconstruct a baseline from a fresh no-memory run; temp-0.6 baseline variance is ~4pp and will flip signs.
3. Report per-type SR alongside the overall (per-type effects anti-correlate across checkpoints — Heat can regress while Clean gains).
4. Report noise-floor-aware conclusions: at n=140, single-arm claims need ~2×SE ≈ 8pp to clear the noise gate.

## checkpoint selection

Never ship `checkpoint-60`. The training trajectory is bimodal on this stack (see `docs/repro_report.md` §4). Sweep every saved intermediate, report the shape, ship the best-on-heldout.

## running training

Two working paths on 8×H100:

```bash
./run_algo1_fft.sh              # FFT + ZeRO-3, ~70 min/step, 60 steps
./run_algo1_v8_lora_kl.sh       # LoRA r=32, ~40 min/step
```

Both require `belt login` for the inference.sh executor + judge, and `alfworld-download` for ALFWorld data. See README.md § Quick start.

## code style

- Configs live in `configs/`. Superseded configs move to `configs/legacy/`. Don't delete — provenance matters.
- Launchers live at the repo root. Superseded launchers move to `legacy/`.
- Test scripts and dev probes live in `scripts/`. Kill them once they've paid rent.
- Prefer editing existing files over adding parallel ones. If a concept already exists (task-type taxonomy, format_trajectory, sweep supervisor), reuse.

## dataset access

- ALFWorld: `alfworld-download -f`, no auth.
- AIME24, AIME25: HuggingFace public.
- GPQA-Diamond: gated. `huggingface-cli login` and request access at https://huggingface.co/datasets/Idavidrein/gpqa. Access is conditional on **not leaking problem text, options, or answers into any public-visible artifact** (git, PRs, docs, screenshots). Report aggregate accuracy only.

## reporting bugs and reproductions

Please include:
- The exact commit hash you ran against.
- The config file used (paste the full YAML if it deviates from a tracked config).
- The JSONL output path (or a link to a gist / uploaded artifact).
- The paired-McNemar comparator output vs the canonical baseline, not summary numbers.

## what NOT to add

- New abstractions without a demonstrated use case. Three similar lines beat a premature helper.
- Unrelated cleanup mixed into a results-affecting change. Split it.
- Emojis in any file. Anthropic-format skills (YAML frontmatter markdown) only.
