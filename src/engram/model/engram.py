"""Engram: encoder + decoder over one shared frozen base model."""

import torch
from torch import nn

from engram.config import Config
from engram.model.base import add_lora
from engram.model.decoder import Decoder
from engram.model.encoder import Encoder
from engram.model.memory import ControlEmbedding
from engram.tokens import add_control_tokens


class Engram(nn.Module):
    def __init__(self, base_model, tokenizer, cfg: Config):
        super().__init__()
        self.cfg, self.tokenizer = cfg, tokenizer
        self.control_ids = add_control_tokens(tokenizer, base_model)
        emb = base_model.get_input_embeddings().weight
        self.hidden_size = emb.shape[1]
        embed_rms = float(emb.pow(2).mean().sqrt())
        self.eos_id = tokenizer.eos_token_id
        self.lm = add_lora(base_model)
        ctrl_init = torch.randn(len(self.control_ids), self.hidden_size) * embed_rms
        self.control = ControlEmbedding(self.control_ids, ctrl_init)
        self.encoder = Encoder(self.lm, cfg.chunk_size, cfg.k, self.hidden_size, embed_rms)
        self.decoder = Decoder(self.lm, self.control)
        self.to(cfg.device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids.to(self.cfg.device))

    def _control(self, density: str, batch: int) -> torch.Tensor:
        tid = self.control_ids[f"<{density}>"]
        return torch.full((batch, 1), tid, dtype=torch.long, device=self.cfg.device)

    def loss(self, chunk_ids: torch.Tensor, density: str = "full") -> torch.Tensor:
        """Reconstruction loss: encode chunks, decode them back followed by EOS."""
        chunk_ids = chunk_ids.to(self.cfg.device)
        b = chunk_ids.shape[0]
        memory = self.encode(chunk_ids)
        eos = torch.full((b, 1), self.eos_id, dtype=torch.long, device=self.cfg.device)
        target = torch.cat([chunk_ids, eos], dim=1)
        return self.decoder.loss(memory, self._control(density, b), target)

    def terse_loss(self, examples: list[dict]) -> torch.Tensor:
        """Summary loss over examples {chunks [n_i, C], target [T_i], density}; memory and
        targets are right-padded and masked so variable chunk counts batch together."""
        dev, k = self.cfg.device, self.cfg.k
        counts = [len(e["chunks"]) for e in examples]
        flat = torch.cat([e["chunks"] for e in examples]).to(dev)
        mem = self.encode(flat.reshape(1, -1))[0].reshape(len(flat), k, -1)
        b, m = len(examples), max(counts) * k
        memory = mem.new_zeros(b, m, mem.shape[-1])
        mmask = torch.zeros(b, m, dtype=torch.long, device=dev)
        t = max(len(e["target"]) for e in examples)
        target = torch.full((b, t), self.eos_id, dtype=torch.long, device=dev)
        tmask = torch.zeros(b, t, dtype=torch.long, device=dev)
        start = 0
        for i, (e, n) in enumerate(zip(examples, counts)):
            memory[i, : n * k] = mem[start : start + n].reshape(n * k, -1)
            mmask[i, : n * k] = 1
            start += n
            target[i, : len(e["target"])] = e["target"].to(dev)
            tmask[i, : len(e["target"])] = 1
        density = examples[0]["density"]
        return self.decoder.loss(memory, self._control(density, b), target, mmask, tmask)

    def tokenize_chunks(self, text: str) -> torch.Tensor:
        """Token ids [1, N*chunk_size], right-padded with EOS to a whole number of chunks."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        c = self.cfg.chunk_size
        ids = ids + [self.eos_id] * (-len(ids) % c or (c if not ids else 0))
        return torch.tensor([ids])

    @torch.no_grad()
    def generate(self, text: str, density: str = "full", max_new_tokens: int | None = None) -> str:
        memory = self.encode(self.tokenize_chunks(text))
        n = max_new_tokens or memory.shape[1] // self.cfg.k * self.cfg.chunk_size + 1
        out = self.decoder.generate(memory, self._control(density, 1), n, self.eos_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
