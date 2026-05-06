"""Disk cache for successful Google Routes / Route Matrix HTTP responses.

**Purpose:** cut API cost, speed up retries, and make integration tests faster when the same
requests repeat. Only **HTTP 200** bodies are stored (errors are never cached).

**Version / invalidation:** bump :data:`ROUTING_CACHE_INTEGRATION_VERSION` whenever request
shape, field masks, URL endpoints, or response parsing logic changes in a way that would make
old cached payloads unsafe to reuse. The cache directory layout includes this version, so older
trees are ignored automatically.

**Environment**

* ``TRAVELPLANNER_ROUTING_CACHE_DIR`` — root directory; defaults to ``~/.cache/travelplanner/routing``.
* ``TRAVELPLANNER_ROUTING_CACHE_DISABLE`` — if ``1`` / ``true``, skip read and write (unit tests, CI).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Bump when changing cache key inputs, endpoints, masks, or anything that invalidates stored JSON.
ROUTING_CACHE_INTEGRATION_VERSION = "1"

_CACHE_HITS = 0
_CACHE_MISSES = 0


def reset_routing_cache_metrics() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def routing_cache_metrics() -> tuple[int, int]:
    return _CACHE_HITS, _CACHE_MISSES


def routing_cache_disabled() -> bool:
    v = os.environ.get("TRAVELPLANNER_ROUTING_CACHE_DISABLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def routing_cache_version_dir() -> Path:
    root = os.environ.get("TRAVELPLANNER_ROUTING_CACHE_DIR", "").strip()
    if root:
        base = Path(root)
    else:
        base = Path.home() / ".cache" / "travelplanner" / "routing"
    return base / f"iv{ROUTING_CACHE_INTEGRATION_VERSION}"


def _key_salt(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:16]


def cache_key(*, kind: str, payload: dict[str, Any], extra: str, api_key: str) -> str:
    """Stable SHA256 filename fragment (hex, no extension)."""
    blob = json.dumps(
        {"kind": kind, "payload": payload, "extra": extra, "salt": _key_salt(api_key)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_cached_body(*, key_hex: str) -> str | None:
    global _CACHE_HITS
    if routing_cache_disabled():
        return None
    path = routing_cache_version_dir() / f"{key_hex}.json"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    _CACHE_HITS += 1
    return text


def write_cached_body(*, key_hex: str, body_text: str) -> None:
    if routing_cache_disabled():
        return
    d = routing_cache_version_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key_hex}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(body_text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def record_cache_miss() -> None:
    global _CACHE_MISSES
    _CACHE_MISSES += 1
