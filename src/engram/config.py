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
    # data
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    data_cache: str = "data/cache"
    train_samples: int = 6400  # 128-token chunks
    val_samples: int = 64
    # training
    lr: float = 2e-3
    steps: int = 800  # optimizer steps
    batch_size: int = 4
    grad_accum: int = 2
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    val_interval: int = 100
    ckpt_interval: int = 200
    num_threads: int = 6  # physical cores; used when device is cpu
    resume: str = ""  # path to a checkpoint .pt to resume from
    # Stage 2 (defaults keep Stage 1 behaviour)
    init_from: str = ""  # checkpoint .pt to load params from (fresh optimizer and step)
    encoder_lr_scale: float = 1.0  # LR multiplier for encoder LoRA + memory queries + projection
    full_ratio: float = 1.0  # fraction of micro-batches that are <full> reconstruction
    terse_dataset: str = "EdinburghNLP/xsum"
    terse_train_samples: int = 0
    terse_val_samples: int = 0
    max_chunks: int = 4
    max_summary_tokens: int = 64


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    return replace(Config(), **data)
