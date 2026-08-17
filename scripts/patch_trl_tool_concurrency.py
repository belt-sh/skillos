"""Make TRL execute one iteration's async tool calls across ALL completions at once.

THE PROBLEM. TRL's tool-calling loop is shaped like this:

    while idxs_with_tool and iteration_num < max_tool_calling_iterations:
        for idx in range(len(idxs_with_tool)):          # one completion at a time
            for tool_call in tool_call_list:
                async_coros.append(coro)                # created, not awaited
            if async_coros:
                asyncio.gather(*async_coros)            # gathered INSIDE the idx loop

The gather only parallelises multiple tool calls *within a single completion*. Our
curator emits exactly one `curate_and_advance` per turn, so it gathers a single
coroutine and the outer loop walks completions serially.

Each of those calls is a full remote ALFWorld episode. So at an effective batch of
32 spread over 8 ranks, only 8 episodes are ever in flight instead of 32, and we
measured ~6. The endpoint is not the bottleneck: a call takes 18.3s under live
training load, and the run was managing 20.6 calls/min, which is ~6 slots.

Our own env carries a comment asserting that TRL's gather "can interleave all 16
rollouts on a rank concurrently". It cannot. `asyncio.to_thread` in the env is
correct and necessary, but there is nothing else queued on the loop to interleave
with.

THE FIX. Split the single pass into three:

    pass 1  per completion: run sync tools, collect async coroutines (no await)
    pass 2  ONE gather across every collected coroutine
    pass 3  per completion: append tool messages, in the original order

Ordering is preserved because pass 3 walks `idx` in the same sequence and reads
results keyed by idx. Sync tools keep their existing semantics. Exceptions keep
being converted to `{"error": ...}` per call, so one failing episode cannot take
down the batch.

Expected effect: ~4x concurrency, so ~4x less wall clock, with no change to any
hyperparameter, the reward, or the data.

This edits the installed TRL in the venv, which is why it lives in the repo as an
idempotent script rather than a manual step: run it after any `pip install trl`
and it either applies cleanly, reports it is already applied, or refuses. It never
half-applies.

    python -m scripts.patch_trl_tool_concurrency          # apply
    python -m scripts.patch_trl_tool_concurrency --check  # report status only
    python -m scripts.patch_trl_tool_concurrency --revert # restore from .orig
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

MARKER = "# ---- skillos: batched async tool execution ----"

# The exact source we replace. If TRL changes this region, the patch refuses
# rather than guessing, and the run keeps its old (correct but slow) behaviour.
OLD = '''            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
'''

NEW = '''            # ---- skillos: batched async tool execution ----
            # Upstream ran the gather inside this loop, so async tools executed
            # one completion at a time. When a tool call is a whole remote
            # episode, that caps concurrency at one episode per rank. Collect
            # first, gather once, then apply results in order.
            # Applied by scripts/patch_trl_tool_concurrency.py.
            _sk_pending = {}      # idx -> list[(name, coro)]
            _sk_sync = {}         # idx -> list[(name, result)]
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
'''

# Second edit: the gather block becomes a deferral, and a new block after the
# loop does the single gather plus the ordered application.
OLD2 = '''                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:'''

NEW2 = '''                # skillos: defer instead of gathering here (see patch script).
                if async_coros:
                    _sk_pending[idx] = list(async_coros)
                _sk_sync[idx] = list(tool_call_results)

            # skillos: ONE gather across every completion's async tool calls.
            _sk_flat = [(idx, name, coro)
                        for idx, pairs in _sk_pending.items()
                        for name, coro in pairs]
            _sk_done = {}
            if _sk_flat:

                async def _sk_run_all(items):
                    return await asyncio.gather(*[c for _, _, c in items],
                                                return_exceptions=True)

                _sk_results = asyncio.run_coroutine_threadsafe(
                    _sk_run_all(_sk_flat), self.async_loop
                ).result()
                for (idx, name, _), result in zip(_sk_flat, _sk_results):
                    _sk_done.setdefault(idx, []).append((name, result))

            # skillos: apply results per completion, in the original order.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                tool_call_results = list(_sk_sync.get(idx, []))
                for name, result in _sk_done.get(idx, []):
                    if isinstance(result, Exception):
                        tool_failure_count += 1
                        tool_call_results.append((name, {"error": str(result)}))
                    else:
                        tool_call_results.append((name, result))

                for name, result in tool_call_results:'''


def target() -> Path:
    import trl
    return Path(trl.__file__).parent / "trainer" / "grpo_trainer.py"


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    path = target()
    src = path.read_text()

    if mode == "--check":
        print(f"{path}\n  patched: {MARKER in src}")
        return 0

    if mode == "--revert":
        orig = path.with_suffix(".py.skillos-orig")
        if not orig.exists():
            print(f"no backup at {orig}"); return 1
        shutil.copy2(orig, path)
        print(f"reverted {path} from {orig}")
        return 0

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    missing = [n for n, s in (("OLD", OLD), ("OLD2", OLD2)) if s not in src]
    if missing:
        print(f"REFUSING: {', '.join(missing)} not found verbatim in {path}.\n"
              f"TRL's tool loop has changed. Re-derive the patch against the new "
              f"source rather than forcing it; a half-applied patch here would "
              f"silently reorder tool results.")
        return 2
    if src.count(OLD) != 1 or src.count(OLD2) != 1:
        print("REFUSING: anchor text is not unique; refusing to guess."); return 2

    backup = path.with_suffix(".py.skillos-orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backed up -> {backup}")

    patched = src.replace(OLD, NEW, 1).replace(OLD2, NEW2, 1)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"REFUSING: patched source does not parse ({e}); nothing written")
        return 3

    path.write_text(patched)
    print(f"patched {path}")
    print("  async tool calls now gather across all completions in one pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
