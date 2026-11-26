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
    # LoRA (encoder and decoder adapters share r/alpha/targets)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # False: the decoder is the frozen base model, so it can only lower the loss via memory
    decoder_lora: bool = True
    # >0: low-rank k/v deltas applied at memory positions only in every decoder layer (reads
    # memory better, can't memorise targets). 0 = off.
    decoder_memory_adapter_rank: int = 0
    # True: a learned vector per control flag added to every decoder layer's residual stream at
    # control + target positions, so the density flag is present at every step (not one token).
    control_vectors: bool = False
    # subtract a running per-slot mean from encoder hidden states before projecting to memory
    center_memory: bool = False
    # memory slot i += RMS-scaled mean input embedding of chunk tokens [i*C/K, (i+1)*C/K)
    memory_segment_residual: bool = False
    # encoder hidden_states index for memory (-1 = final normed output; SmolLM2 has 30 layers)
    memory_layer: int = -1
    # hidden dims zeroed before the memory projection (SmolLM2 massive activations: 17, 507)
    memory_mask_dims: tuple[int, ...] = ()
    # memory slot i += proj(mean encoder hidden state over chunk tokens [i*C/K, (i+1)*C/K))
    memory_pooled_residual: bool = False
    # fraction of decoder teacher-forcing inputs replaced by EOS in training (word dropout)
    decoder_input_dropout: float = 0.0
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
    multi_full_ratio: float = 0.0  # fraction of <full> micro-batches that rebuild a whole multi-chunk article
    terse_dataset: str = "EdinburghNLP/xsum"
    terse_train_samples: int = 0
    terse_val_samples: int = 0
    max_chunks: int = 4
    max_summary_tokens: int = 64
    # Stage 3: fraction of all micro-batches that are <answer> (SQuAD); full_ratio splits the rest
    answer_ratio: float = 0.0
    qa_dataset: str = "rajpurkar/squad"
    qa_train_samples: int = 0
    qa_val_samples: int = 0
    qa_eval_samples: int = 200  # held-out SQuAD dev subset for scripts/eval.py --qa
    qa_eval_seed: int = 0  # seeds the shuffled-memory pairing
    # Stage 3 validation: QA F1 (real/shuffled/zeroed) on the first N answer_val examples and density
    # on the first N terse_val articles, generated at every val step; 0 = off (generation is slow)
    val_gen_samples: int = 0
    val_max_full_tokens: int = 0  # cap on <full> generation during val density; 0 = article length
    # Stage 3b warm-up: fraction of reconstruction train chunks drawn from SQuAD train paragraphs
    # (needs qa_train_samples examples to draw from; 0 = WikiText only)
    squad_recon_share: float = 0.0
    # Stage 3b joint: fraction of <full> micro-batches that rebuild a whole SQuAD train paragraph
    # (distinct paragraphs of the answer_train examples); multi_full_ratio splits the rest
    squad_full_ratio: float = 0.0
    # validation: <full> answer recall on the first N answer_val examples (needs answer_val; 0 = off)
    val_recall_samples: int = 0
    # Text-QA control (scripts/train_text_qa.py): frozen base + one LoRA (lora_r, lr) trained on
    # `paragraph\nQuestion: q\nAnswer:` -> answer. Optimizer steps, each grad_accum micro-batches of
    # batch_size; set to Stage 3's answer_ratio * steps so both see the same number of QA examples.
    text_qa_steps: int = 0
    # cached control {em, f1}; scripts/eval.py --qa reports memory F1 / control F1 when it exists
    text_qa_result: str = "runs/text_qa_control.json"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    for key in ("lora_targets", "memory_mask_dims"):
        if key in data:
            data[key] = tuple(data[key])
    return replace(Config(), **data)
