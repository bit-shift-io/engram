import pytest

from engram.eval.density import aggregate_density, strip_trailing_eos


def rec(full, terse, t_rouge, b_rouge):
    return {"full_len": full, "terse_len": terse, "terse_rouge_l": t_rouge, "baseline_rouge_l": b_rouge}


def test_aggregate_length_ratio_and_rouge():
    out = aggregate_density([rec(100, 20, 0.4, 0.2), rec(50, 25, 0.2, 0.1)])
    assert out["n"] == 2
    assert out["mean_full_len"] == 75
    assert out["mean_terse_len"] == 22.5
    assert out["length_ratio"] == pytest.approx((0.2 + 0.5) / 2)
    assert out["terse_rouge_l"] == pytest.approx(0.3)
    assert out["baseline_rouge_l"] == pytest.approx(0.15)


def test_aggregate_skips_zero_length_full_in_ratio():
    out = aggregate_density([rec(0, 5, 0, 0), rec(10, 5, 0, 0)])
    assert out["length_ratio"] == 0.5


def test_strip_trailing_eos():
    assert strip_trailing_eos([1, 2, 9, 9], 9) == [1, 2]
    assert strip_trailing_eos([9, 9], 9) == []
