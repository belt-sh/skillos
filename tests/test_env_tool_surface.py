"""Only `curate_and_advance` may reach the curator as a callable tool.

TRL builds the tool list from every public method on the environment:

    for name, member in inspect.getmembers(environment, predicate=inspect.ismethod):
        if name == "reset":
            has_reset = True
        elif not name.startswith("_"):
            environment_methods[i].append(member)
    # trl/trainer/grpo_trainer.py:501-504

So adding a public helper to Algo1CuratorEnv silently adds a tool. On 2026-08-18
`complete_unplayed_positions` was added as a public method and TRL tried to
publish it, dying ten minutes into the run while generating its JSON schema
(`DocstringParsingException: no description for the argument 'deadline'`).

The crash was luck. That helper finishes the rollout's remaining positions; had
its docstring satisfied the schema generator, the curator would have been handed
a tool that completes its own protocol, and the run would have trained against a
silently larger action space. This test is cheaper than that discovery.

Run: .venv/bin/python tests/test_env_tool_surface.py
"""
from __future__ import annotations

import inspect
import sys

EXPECTED_TOOLS = {"curate_and_advance"}


def discovered_tools(env) -> set[str]:
    """Replicate TRL's rule exactly, including its special case for `reset`."""
    tools = set()
    has_reset = False
    for name, member in inspect.getmembers(env, predicate=inspect.ismethod):
        if name == "reset":
            has_reset = True
        elif not name.startswith("_"):
            tools.add(name)
    assert has_reset, "TRL requires a callable `reset` on every environment"
    return tools


def main() -> int:
    from skillos.algo1 import Algo1CuratorEnv

    env = Algo1CuratorEnv()
    tools = discovered_tools(env)

    failures = []
    unexpected = tools - EXPECTED_TOOLS
    if unexpected:
        failures.append(
            f"these public methods would be published to the curator as tools: "
            f"{sorted(unexpected)}. Prefix them with '_' unless the model is "
            f"meant to call them.")
    missing = EXPECTED_TOOLS - tools
    if missing:
        failures.append(f"expected tools are not exposed: {sorted(missing)}")

    # Every exposed tool must also survive schema generation, which is what
    # actually crashed the run.
    if not failures:
        from transformers.utils.chat_template_utils import get_json_schema
        for name in sorted(tools):
            try:
                get_json_schema(getattr(env, name))
            except Exception as exc:
                failures.append(f"{name}: schema generation fails: "
                                f"{type(exc).__name__}: {exc}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"OK: exactly {sorted(tools)} exposed, schema generates cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
