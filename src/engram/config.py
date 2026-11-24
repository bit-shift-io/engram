"""YAML -> dataclass config."""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    device: str = "cpu"
    chunk_size: int = 128
    k: int = 16
    seed: int = 0
    run_dir: str = "runs"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    return replace(Config(), **data)
