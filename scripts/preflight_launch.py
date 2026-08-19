#!/usr/bin/env python3
"""Gate every training launch on the checks that historically cost GPU-days.

WHY THIS EXISTS
---------------
The reproduction's largest category of wasted compute was not crashes. It was
runs that started, looked healthy, and were later found to have been training
against something wrong. Each check below names the incident it would have
caught, and every one of them is cheap, CPU-only, and answerable before a single
GPU-second is spent:

  ~64 GPU-days  three runs on a degenerate task distribution
  ~10 box-days  a run at 2x the paper's batch (justified by a comment that did
                not bind the value)
  11 weeks      max_completion_length capping |G|=10 at ~3 positions, because a
                comment beside it named the paper's hyperparameter
  first 24h     an executor crippled by max_tokens=256 with reasoning off, its
                own config comment already saying to change it
  ~21h          r_task divided by positions played, rewarding early exit
  10 min x3     a public env method published to the model as a tool

Run standalone or via run_algo1_dense.sh, which aborts on any FAIL.

    .venv/bin/python scripts/preflight_launch.py configs/alfworld_dense_fft.yaml

Exit 0 = every check passed. Exit 1 = at least one FAIL. Exit 2 = could not run.
"""
from __future__ import annotations

import inspect
import os
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# Paper Table 4, ALFWorld column, transcribed from docs/skillos_paper.md:385-400.
# Values the paper states directly. Anything we deliberately depart from must be
# listed in SANCTIONED with a reason, so a silent drift cannot masquerade as one.
PAPER_ALFWORLD = {
    "learning_rate": 1.0e-6,
    "effective_batch": 32,
    "beta": 0.001,
    "num_generations": 8,       # "GRPO group size"
    "temperature": 1.0,
    "max_steps": 60,            # "Steps"
    "group_size": 10,           # "Data Grouping Size"
}
PAPER_EXECUTOR = {
    "history_length": 3,        # "Action history length"
    "max_turns": 30,            # "Max number of turns"
    "top_k_retrieval": 5,       # "Top-K skill retrieval"
}

# Deliberate, documented departures. Key -> (expected value, why).
SANCTIONED = {
    "max_completion_length": (
        16384,
        "Paper's 4,096 is a cap on the curator's RESPONSE. TRL applies it to the "
        "ACCUMULATED multi-turn completion (grpo_trainer.py:1620), so 4096 holds "
        "~3 of 10 positions. Raising it is more faithful, not less. "
        "See DIVERGENCES #16.",
    ),
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))


def main(argv: list[str]) -> int:
    cfg_path = Path(argv[1]) if len(argv) > 1 else REPO / "configs/alfworld_dense_fft.yaml"
    if not cfg_path.exists():
        print(f"preflight: cannot read {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    acc_path = REPO / "configs/accelerate_zero3.yaml"
    acc = yaml.safe_load(acc_path.read_text()) if acc_path.exists() else {}
    ds = (acc.get("deepspeed_config") or {})
    world = int(acc.get("num_processes", 8))

    # 1. Paper hyperparameters, one line each.
    for key, want in PAPER_ALFWORLD.items():
        if key == "effective_batch":
            got = (int(cfg.get("batch_size", 1))
                   * int(cfg.get("gradient_accumulation_steps", 1)) * world)
        else:
            got = cfg.get(key)
        same = (abs(got - want) < 1e-12 if isinstance(want, float) and got is not None
                else got == want)
        check(f"paper/{key}", same, f"want {want}, config has {got}")

    # 2. Sanctioned departures must be exactly as sanctioned.
    for key, (want, why) in SANCTIONED.items():
        got = cfg.get(key)
        check(f"sanctioned/{key}", got == want,
              f"want {want} ({why[:60]}...), config has {got}")

    # 3. Batch arithmetic must agree across BOTH config files, or DeepSpeed
    #    aborts all ranks at launch (2026-08-17).
    check("batch/deepspeed_accum_matches",
          int(ds.get("gradient_accumulation_steps", -1))
          == int(cfg.get("gradient_accumulation_steps", -2)),
          f"training config {cfg.get('gradient_accumulation_steps')} vs "
          f"accelerate {ds.get('gradient_accumulation_steps')}")

    # 4. Generation concurrency must be pinned, not left to track accum
    #    (DIVERGENCES #17a).
    gb = cfg.get("generation_batch_size")
    ng = int(cfg.get("num_generations", 8))
    check("batch/generation_pinned", gb is not None,
          "generation_batch_size unset: concurrency will silently track accum")
    if gb is not None:
        check("batch/generation_divisible", int(gb) % ng == 0,
              f"generation_batch_size {gb} must be divisible by num_generations {ng}")

    # 5. The executor must not be crippled. Its own config comment once said to
    #    change these and the run went ahead anyway (2026-05-20).
    ex = cfg.get("executor") or {}
    check("executor/reasoning_on", str(ex.get("reasoning_effort")) not in ("none", "None"),
          f"reasoning_effort={ex.get('reasoning_effort')!r} disables CoT the prompt requires")
    check("executor/max_tokens_sane", int(ex.get("max_tokens", 0)) >= 4096,
          f"max_tokens={ex.get('max_tokens')} truncates reasoning (256 cost the first 24h)")
    check("executor/history_length", int(ex.get("history_length", -1))
          == PAPER_EXECUTOR["history_length"],
          f"paper says {PAPER_EXECUTOR['history_length']}, config has {ex.get('history_length')}")
    steps_env = int(os.environ.get("SKILLOS_EXECUTOR_MAX_STEPS", "0") or 0)
    check("executor/max_turns", steps_env >= PAPER_EXECUTOR["max_turns"],
          f"SKILLOS_EXECUTOR_MAX_STEPS={steps_env or 'unset'}, paper caps turns at "
          f"{PAPER_EXECUTOR['max_turns']} (default 10 truncates long-horizon tasks)")

    # 6. Crash economics: a step is ~2h, so save_steps must be 1, and on resume
    #    transformers restores save_steps from the checkpoint (cost ~15h in June).
    check("io/save_steps", int(cfg.get("save_steps", 0)) == 1,
          f"save_steps={cfg.get('save_steps')}; at ~2h/step anything higher "
          f"discards hours on a crash")
    out = REPO / str(cfg.get("output_dir", "")).lstrip("./")
    latest = sorted(out.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1])) if out.exists() else []
    if latest:
        import json
        st = latest[-1] / "trainer_state.json"
        got = json.loads(st.read_text()).get("save_steps") if st.exists() else None
        check("io/resume_save_steps", got in (None, 1),
              f"{latest[-1].name} carries save_steps={got}; transformers restores "
              f"it over the config on resume")

    # 7. Disk, on the RESOLVED path. `df ~` reports the wrong filesystem when
    #    output/ is a symlink, which produced a false alarm on 2026-08-18.
    probe = out if out.exists() else REPO / "output"
    try:
        free_gb = shutil.disk_usage(probe.resolve()).free / 2**30
        need = 12 * 107  # save_total_limit x measured ZeRO-3 checkpoint size
        check("io/disk_free", free_gb > need,
              f"{free_gb:.0f} GB free on {probe.resolve()}, need ~{need} GB for "
              f"{cfg.get('save_total_limit')} checkpoints at ~107 GB")
    except Exception as exc:
        check("io/disk_free", False, f"could not stat {probe}: {exc}")

    # 8. Tool surface: any public env method is published to the model
    #    (grpo_trainer.py:501-504). Cost three launches on 2026-08-18.
    try:
        sys.path.insert(0, str(REPO))
        from skillos.algo1 import Algo1CuratorEnv
        from transformers.utils.chat_template_utils import get_json_schema
        env = Algo1CuratorEnv()
        tools, has_reset = set(), False
        for n, m in inspect.getmembers(env, predicate=inspect.ismethod):
            if n == "reset":
                has_reset = True
            elif not n.startswith("_"):
                tools.add(n)
        check("env/reset_present", has_reset, "TRL requires a callable reset()")
        check("env/tool_surface", tools == {"curate_and_advance"},
              f"exposed to the model: {sorted(tools)}")
        schema_err = ""
        for n in sorted(tools):
            try:
                get_json_schema(getattr(env, n))
            except Exception as exc:
                schema_err += f"{n}: {type(exc).__name__}; "
        check("env/tool_schemas", not schema_err, schema_err or "all generate")
    except Exception as exc:
        check("env/tool_surface", False, f"could not import env: {type(exc).__name__}: {exc}")

    # 9. r_task arithmetic: quitting early must never beat playing it out
    #    (DIVERGENCES #18). Re-derived here so the gate does not depend on pytest.
    try:
        from skillos.algo1 import env as A
        A.configure(judge_submit=None, num_generations=8,
                    group_size=int(cfg.get("group_size", 10)))

        def r_task(played_successes: int, played: int, infra_lost: int) -> float:
            informed = A._group_size - 1
            denom = informed - infra_lost
            return played_successes / denom if denom > 0 else float("nan")

        early, honest = r_task(1, 1, 0), r_task(4, 9, 0)
        check("reward/early_exit_not_rewarded", early < honest,
              f"one success then quit scores {early:.3f}, nine positions with four "
              f"successes scores {honest:.3f}")
        check("reward/infra_leaves_denominator",
              abs(r_task(1, 1, 8) - 1.0) < 1e-9,
              "positions lost to infrastructure must leave the denominator")
    except Exception as exc:
        check("reward/arithmetic", False, f"{type(exc).__name__}: {exc}")

    # Report.
    width = max(len(n) for n, _, _ in results)
    failed = [r for r in results if not r[1]]
    print(f"preflight: {cfg_path.name} against paper Table 4 "
          f"(docs/skillos_paper.md:385-400)\n")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}".rstrip())
    print()
    if failed:
        print(f"preflight FAILED: {len(failed)} of {len(results)} checks. "
              f"Not launching.", file=sys.stderr)
        return 1
    print(f"preflight OK: {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
