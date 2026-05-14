"""Smoke test — verify all dependencies load and basic components work."""

import sys


def main():
    errors = []

    # 1. TRL
    print("Checking TRL...", end=" ")
    try:
        from trl import GRPOConfig, GRPOTrainer
        print("OK")
    except ImportError as e:
        print(f"FAIL: {e}")
        errors.append("trl")

    # 2. Transformers + PEFT
    print("Checking transformers + peft...", end=" ")
    try:
        import transformers
        import peft
        print("OK")
    except ImportError as e:
        print(f"FAIL: {e}")
        errors.append("transformers/peft")

    # 3. vLLM (optional — only needed on GPU)
    print("Checking vLLM...", end=" ")
    try:
        import vllm
        print("OK")
    except ImportError:
        print("SKIP (optional, install with pip install skillos[gpu])")

    # 4. ALFWorld
    print("Checking ALFWorld...", end=" ")
    try:
        import alfworld
        print("OK")
    except ImportError as e:
        print(f"FAIL: {e}")
        errors.append("alfworld")

    # 5. rank-bm25
    print("Checking rank-bm25...", end=" ")
    try:
        from rank_bm25 import BM25Okapi
        print("OK")
    except ImportError as e:
        print(f"FAIL: {e}")
        errors.append("rank-bm25")

    # 6. SkillRepo
    print("Checking SkillRepo...", end=" ")
    try:
        from skillos.skills.repo import SkillRepo
        repo = SkillRepo()
        repo.insert("test-skill", "---\nname: Test\ndescription: A test skill\n---\n# Workflow\n1. Do the thing")
        results = repo.retrieve("test", top_k=1)
        assert len(results) == 1
        assert results[0].name == "test-skill"
        assert repo.delete("test-skill")
        assert len(repo) == 0
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append("skillos.skills.repo")

    # 7. Rewards
    print("Checking rewards...", end=" ")
    try:
        from skillos.rewards.composite import composite_reward
        r = composite_reward(r_task=0.5, r_fc=1.0, r_cnt=0.8, r_comp=0.9)
        assert r > 0
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append("skillos.rewards")

    # 8. Prompts
    print("Checking prompts...", end=" ")
    try:
        from skillos.curator.prompts import CURATOR_SYSTEM, CURATOR_TOOLS
        assert len(CURATOR_TOOLS) == 3
        assert "new_skill_insert" in CURATOR_TOOLS[0]["function"]["name"]
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append("skillos.curator.prompts")

    # 9. CUDA
    print("Checking CUDA...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"OK ({device}, {mem:.0f}GB)")
        else:
            print("SKIP (CPU only — training will use accelerate CPU mode)")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append("cuda")

    print()
    if errors:
        print(f"FAILED: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
