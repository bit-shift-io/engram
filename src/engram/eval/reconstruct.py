"""Greedy reconstruction accuracy and ROUGE-L under memory conditions."""

import torch

from engram.eval.ablation import CONDITIONS


def token_accuracy(pred: list[int], target: list[int]) -> float:
    """Position-wise match rate over the target length."""
    return sum(a == b for a, b in zip(pred, target)) / len(target)


def rouge_l(pred: list[int], target: list[int]) -> float:
    """ROUGE-L F1 over token ids via longest common subsequence."""
    if not pred or not target:
        return 0.0
    prev = [0] * (len(target) + 1)
    for a in pred:
        cur = [0]
        for j, b in enumerate(target):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(target)
    return 2 * p * r / (p + r)


@torch.no_grad()
def evaluate_reconstruction(eng, chunks: torch.Tensor, batch_size: int = 8) -> dict:
    eng.eval()
    cfg = eng.cfg
    ctrl = eng._control("full", 1)
    mems = [eng.encode(chunks[i : i + batch_size]) for i in range(0, len(chunks), batch_size)]
    out = {"n_chunks": len(chunks), "chunk_size": cfg.chunk_size, "k": cfg.k,
           "compression_ratio": cfg.chunk_size / cfg.k}
    for name, fn in CONDITIONS.items():
        accs, rouges = [], []
        i = 0
        for m in mems:
            m = fn(m)
            for row in range(m.shape[0]):
                target = chunks[i].tolist()
                pred = eng.decoder.generate(m[row : row + 1], ctrl, len(target) + 1, eng.eos_id)
                accs.append(token_accuracy(pred, target))
                rouges.append(rouge_l(pred, target))
                i += 1
        out[name] = {"token_accuracy": sum(accs) / len(accs), "rouge_l": sum(rouges) / len(rouges)}
    return out
