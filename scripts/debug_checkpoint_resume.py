"""End-to-end test of the SkillOS checkpoint save+resume path.

Verifies:
  - SkillRepoSaver writes skills/, rollouts.jsonl, judge_cache.json
  - SkillRepoLoader restores the skill repo + judge cache (rollouts not
    intentionally copied back — they're snapshotted for post-crash analytics)
  - Counts and content match across the save/clear/load cycle

Run:
  source .venv/bin/activate && python scripts/debug_checkpoint_resume.py
"""
from __future__ import annotations

import os
import tempfile

import skillos.envs.curator_env as ce
from skillos.skills.repo import SkillRepo


SKILL_BODIES = [
    (
        "open_cabinet",
        "---\nname: open_cabinet\ndescription: open a cabinet door\n---\n"
        "# Open Cabinet\nNavigate to the cabinet, then issue 'open cabinet'.",
    ),
    (
        "place_object",
        "---\nname: place_object\ndescription: place a held object on a surface\n---\n"
        "# Place Object\nApproach the surface, then 'put object on surface'.",
    ),
    (
        "wash_item",
        "---\nname: wash_item\ndescription: wash an item at the sink\n---\n"
        "# Wash Item\nGo to sink, 'use sinkbasin', then 'take item'.",
    ),
]


def _populate_state():
    ce._shared_skill_repo = SkillRepo()
    for name, body in SKILL_BODIES:
        assert ce._shared_skill_repo.insert(name, body), name
    with ce._judge_cache_lock:
        ce._judge_cache.clear()
        ce._judge_cache.update({
            "hash_a": 0.5,
            "hash_b": 1.0,
            "hash_c": 0.25,
        })


def main():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_root = os.path.join(tmp, "checkpoint-7")
        rollouts_src = os.path.join(tmp, "rollouts.jsonl")
        with open(rollouts_src, "w") as f:
            f.write('{"ts": 1, "slot": 0, "reward": 0.5}\n')
            f.write('{"ts": 2, "slot": 1, "reward": 0.7}\n')

        # 1) populate
        _populate_state()
        n_skills_pre = len(ce._shared_skill_repo)
        n_cache_pre = len(ce._judge_cache)
        assert n_skills_pre == 3, n_skills_pre
        assert n_cache_pre == 3, n_cache_pre

        # 2) save (exercises the production save path directly)
        ce.save_curator_state(ckpt_root, rollouts_src)

        # Verify expected files landed on disk
        for fn in ["skills/open_cabinet.md", "skills/place_object.md",
                   "skills/wash_item.md", "rollouts.jsonl", "judge_cache.json"]:
            full = os.path.join(ckpt_root, fn)
            assert os.path.exists(full), f"missing after save: {fn}"

        # 3) wipe in-memory state
        ce._shared_skill_repo = SkillRepo()
        with ce._judge_cache_lock:
            ce._judge_cache.clear()
        assert len(ce._shared_skill_repo) == 0
        assert len(ce._judge_cache) == 0

        # 4) load (exercises the production load path directly)
        ce.load_curator_state(ckpt_root)

        # 5) verify restored
        assert len(ce._shared_skill_repo) == n_skills_pre, (
            f"skill count mismatch: got {len(ce._shared_skill_repo)} "
            f"want {n_skills_pre}"
        )
        for name, body in SKILL_BODIES:
            assert name in ce._shared_skill_repo.skills, f"missing skill: {name}"
            assert ce._shared_skill_repo.skills[name].content == body, name
        assert len(ce._judge_cache) == n_cache_pre, (
            f"judge cache count mismatch: got {len(ce._judge_cache)} "
            f"want {n_cache_pre}"
        )
        for k, v in [("hash_a", 0.5), ("hash_b", 1.0), ("hash_c", 0.25)]:
            assert ce._judge_cache[k] == v, k

        # Rollouts present in the checkpoint (snapshot only — not loaded back)
        with open(f"{ckpt_root}/rollouts.jsonl") as f:
            rollouts_lines = f.readlines()
        assert len(rollouts_lines) == 2

    print("[OK] checkpoint save/load round-trip")
    print(f"     skills restored:     {len(ce._shared_skill_repo)}")
    print(f"     judge cache:         {len(ce._judge_cache)} entries")
    print(f"     rollouts snapshot:   {len(rollouts_lines)} lines (in checkpoint)")


if __name__ == "__main__":
    main()
