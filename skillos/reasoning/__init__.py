"""Reasoning benchmark (AIME24, AIME25, GPQA-Diamond) — SkillOS §4.1/Table 2.

Companion to `skillos/envs/curator_env.py` (ALFWorld). Reuses the same
`SkillRepo`, curator prompts, and streaming-curation loop; swaps the ALFWorld
executor for a CoT-prompted single-turn math/GPQA solver against the same
remote inference.sh executor.
"""
