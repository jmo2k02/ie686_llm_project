from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _load_dotenv_at_module_init() -> None:
    root = Path(__file__).resolve().parents[4]
    load_dotenv(root / ".env")


# Load .env from project root so env vars (e.g. TAVILY_API_KEY) are available
# to all modules that import from travelplanner.config without each file
# having to call load_dotenv() separately.
_ = _load_dotenv_at_module_init()


_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_CACHE_KEY: tuple[str, str] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return payload


def _config_path_from_env(var: str, default_relative: str) -> Path:
    """Resolve YAML paths from env; relative paths are rooted at the repo (not process cwd)."""
    root = _repo_root()
    raw = os.getenv(var)
    if raw is None or not str(raw).strip():
        return (root / default_relative).resolve()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def load_settings(*, force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    global _CONFIG_CACHE_KEY

    global_config_path = _config_path_from_env(
        "TRAVELPLANNER_GLOBAL_CONFIG_PATH",
        "config.yaml",
    )
    local_config_path = _config_path_from_env(
        "TRAVELPLANNER_LOCAL_CONFIG_PATH",
        "local.config.yaml",
    )
    current_key = (str(global_config_path), str(local_config_path))
    if (
        _CONFIG_CACHE is not None
        and not force_reload
        and _CONFIG_CACHE_KEY == current_key
    ):
        return _CONFIG_CACHE

    global_cfg = _load_yaml(global_config_path)
    local_cfg = _load_yaml(local_config_path)
    _CONFIG_CACHE = _deep_merge(global_cfg, local_cfg)
    _CONFIG_CACHE_KEY = current_key
    return _CONFIG_CACHE


def get_setting(path: str, default: Any = None) -> Any:
    current: Any = load_settings()
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
