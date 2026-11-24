import torch


def reconstruction_loss(eng, batch: torch.Tensor) -> torch.Tensor:
    return eng.loss(batch, density="full")


@torch.no_grad()
def validation_loss(eng, chunks: torch.Tensor, batch_size: int) -> float:
    eng.eval()
    total = 0.0
    for i in range(0, len(chunks), batch_size):
        b = chunks[i : i + batch_size]
        total += float(reconstruction_loss(eng, b)) * len(b)
    eng.train()
    return total / len(chunks)
