import pytest
import torch
import yaml
from dataclasses import asdict

from engram.eval.ablation import shuffle_memory, zero_memory
from engram.eval.reconstruct import evaluate_reconstruction, rouge_l, token_accuracy
from engram.eval.report import load_results, markdown_table, write_run_report
from test_overfit import overfit


def test_token_accuracy_and_rouge():
    assert token_accuracy([1, 2, 9, 4], [1, 2, 3, 4]) == 0.75
    assert token_accuracy([1, 2], [1, 2, 3, 4]) == 0.5
    assert rouge_l([1, 2, 3], [1, 2, 3]) == 1.0
    assert rouge_l([], [1]) == 0.0
    assert rouge_l([9, 1, 2], [1, 2, 3]) == pytest.approx(2 / 3)


def test_ablation_transforms():
    m = torch.arange(12.0).view(3, 2, 2)
    assert zero_memory(m).abs().sum() == 0
    s = shuffle_memory(m)
    assert all(not torch.equal(s[i], m[i]) for i in range(3))
    assert sorted(s.view(3, -1).sum(1).tolist()) == sorted(m.view(3, -1).sum(1).tolist())
    with pytest.raises(ValueError):
        shuffle_memory(m[:1])


def test_real_memory_beats_zeroed_and_report(tiny_engram, tmp_path):
    torch.manual_seed(0)
    eng = tiny_engram(chunk_size=16, k=4)
    ids = torch.randint(2, 64, (4, 16))
    overfit(eng, ids, steps=250, lr=1e-2)
    r = evaluate_reconstruction(eng, ids, batch_size=2)
    assert r["compression_ratio"] == 4
    assert r["real"]["token_accuracy"] >= 0.8
    assert r["real"]["token_accuracy"] > r["zeroed"]["token_accuracy"] + 0.4
    assert r["real"]["token_accuracy"] > r["shuffled"]["token_accuracy"] + 0.4
    write_run_report(tmp_path, r)
    assert (tmp_path / "eval" / "reconstruction.md").exists()
    table = markdown_table(load_results([tmp_path, tmp_path]))
    assert table.count("| 4 | 4x |") == 2
