"""config.py

全局配置：
- LLM API key / 模型选择
- 默认阈值参数（电压越限、热稳定越限）

说明：
- Streamlit 部署时建议通过环境变量注入，不要把 key 写进代码库。
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = _strip_wrapping_quotes(value)
    return out


def _load_local_env_defaults() -> None:
    """Load .env/.env.local defaults without overriding process environment."""
    root = Path(__file__).resolve().parent
    merged: dict[str, str] = {}
    merged.update(_parse_env_file(root / ".env"))
    merged.update(_parse_env_file(root / ".env.local"))
    for k, v in merged.items():
        if k not in os.environ:
            os.environ[k] = v


_load_local_env_defaults()


# -----------------------------
# LLM
# -----------------------------

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_TIMEOUT_S: float = float(os.getenv("OPENAI_TIMEOUT_S", "60"))

GEMINI_API_KEY: str | None = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.0")))
GEMINI_TIMEOUT_S: float = float(os.getenv("GEMINI_TIMEOUT_S", os.getenv("LLM_TIMEOUT_S", "60")))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


# MATPOWER text dataset (for blueprint-mode llm_only)
MATPOWER_DATA_ROOT: str = os.getenv("MATPOWER_DATA_ROOT", "data/matpower")
MATPOWER_CASE_DATE: str = os.getenv("MATPOWER_CASE_DATE", "2017-01-01")
LLM_ONLY_DEBUG_MODE: bool = _env_bool("LLM_ONLY_DEBUG_MODE", default=False)


# -----------------------------
# Validation thresholds
# -----------------------------

DEFAULT_V_MIN: float = float(os.getenv("V_MIN", "0.95"))
DEFAULT_V_MAX: float = float(os.getenv("V_MAX", "1.05"))
DEFAULT_MAX_LOADING: float = float(os.getenv("MAX_LOADING", "100"))
