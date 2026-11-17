"""Settings: config.yaml, overridable per key by PDFQA_<KEY> environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = "PDFQA_"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    collection_name: str = "pdf_chunks"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    temperature: float | None = 0.1
    max_answer_tokens: int = 1500
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    search_top_k: int = 6
    search_candidates: int = 30
    agent_max_steps: int = 8
    max_tool_result_tokens: int = 4000
    history_turns: int = 4
    max_upload_mb: int = 50
    max_pages: int = 1000
    request_timeout_s: float = 90
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.chunk_tokens < 50:
            raise ValueError("chunk_tokens must be at least 50")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_tokens // 2:
            raise ValueError("chunk_overlap_tokens must be between 0 and half of chunk_tokens")
        if self.search_top_k < 1 or self.search_candidates < self.search_top_k:
            raise ValueError("need 1 <= search_top_k <= search_candidates")
        if self.agent_max_steps < 2:
            raise ValueError("agent_max_steps must be at least 2 (one tool round + the answer)")


def _coerce(value: str, target: Any) -> Any:
    """Convert an environment-variable string to the type of the default value."""
    if value.lower() in ("null", "none", ""):
        return None
    if isinstance(target, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float) or target is None:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Build Settings from YAML, then environment variables, then explicit overrides."""
    path = config_path or PROJECT_ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    known = {f.name: f for f in fields(Settings)}
    unknown = set(raw) - set(known)
    if unknown:
        logging.getLogger(__name__).warning("Ignoring unknown config keys: %s", sorted(unknown))

    values: dict[str, Any] = {k: v for k, v in raw.items() if k in known}
    for name in known:
        env = os.getenv(ENV_PREFIX + name.upper())
        if env is not None:
            values[name] = _coerce(env, values.get(name, known[name].default))
    values.update(overrides)

    data_dir = Path(values.pop("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    settings = Settings(data_dir=data_dir, **values)
    settings.validate()
    return settings


@lru_cache
def get_settings() -> Settings:
    return load_settings()


def get_api_key() -> str | None:
    """OpenAI key from the environment or .env; None when missing or still the placeholder."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.endswith("your-key-here"):
        return None
    return key


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
