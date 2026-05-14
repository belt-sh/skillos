"""Shared ALFWorld config loading."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_alfworld_config() -> dict:
    config_path = os.environ.get(
        "ALFWORLD_CONFIG",
        str(Path(__file__).parent.parent.parent / "configs" / "alfworld_env.yaml"),
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_alfworld_env(train_eval: str = "train"):
    """Create and initialize an ALFWorld TextWorld environment."""
    from alfworld.agents.environment import get_environment

    config = load_alfworld_config()
    AlfredTWEnv = get_environment("AlfredTWEnv")
    env = AlfredTWEnv(config, train_eval=train_eval)
    return env.init_env(batch_size=1)
