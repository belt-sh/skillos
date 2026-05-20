"""Resolve the inference.sh API key without ever requiring it in source / configs.

Lookup order (first hit wins):
1. explicit `api_key` argument
2. INFSH_API_KEY env var
3. INFERENCESH_API_KEY env var
4. ~/.inferencesh/config.json (`api_key` field; written by `belt login`)
5. .env file at repo root (parsed shallowly; INFSH_API_KEY=...)

Raises a clear error pointing the user at `belt login --key ...` if nothing
resolved. Keys never get logged or echoed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


_BELT_CONFIG = Path.home() / ".inferencesh" / "config.json"


def _from_belt_config() -> str | None:
    try:
        if not _BELT_CONFIG.is_file():
            return None
        data = json.loads(_BELT_CONFIG.read_text())
        key = data.get("api_key")
        return key if isinstance(key, str) and key else None
    except Exception:
        return None


def _from_env_file() -> str | None:
    # Walk up from cwd looking for a .env. Stop after 5 levels.
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents][:6]:
        env_path = parent / ".env"
        if not env_path.is_file():
            continue
        try:
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() in {"INFSH_API_KEY", "INFERENCESH_API_KEY"}:
                    return v.strip().strip("'\"")
        except Exception:
            pass
    return None


def resolve_infsh_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for env_var in ("INFSH_API_KEY", "INFERENCESH_API_KEY"):
        if os.environ.get(env_var):
            return os.environ[env_var]
    key = _from_belt_config()
    if key:
        return key
    key = _from_env_file()
    if key:
        return key
    raise RuntimeError(
        "No inference.sh API key found. Run `belt login --key <KEY>` or set "
        "INFSH_API_KEY in your shell / .env file."
    )
