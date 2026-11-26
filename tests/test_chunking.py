from engram.data.chunking import chunk_tokens
from engram.data.reconstruction import clean_lines


def test_exact_chunks_and_tail_dropped():
    out = chunk_tokens(list(range(10)), 4)
    assert out == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_short_input_gives_no_chunks():
    assert chunk_tokens([1, 2], 4) == []


def test_clean_lines_drops_blank_and_headings():
    lines = ["", " = Title = \n", " Some text here . \n", " = = Sec = = \n", "\n"]
    assert list(clean_lines(lines)) == ["Some text here ."]


def test_mix_chunks_share_of_squad_paragraph_chunks_and_dedupes_paragraphs():
    import torch

    from engram.data.reconstruction import mix_chunks

    wiki = torch.zeros(10, 4, dtype=torch.long)
    para = lambda v: {"chunks": torch.full((2, 4), v)}  # noqa: E731
    squad = [para(1), para(1), para(2), para(3)]  # paragraph 1 repeated per question
    mixed = mix_chunks(wiki, squad, 0.4)
    assert mixed.shape == (10, 4)
    assert (mixed[:, 0] == 0).sum() == 6
    assert set(mixed[:, 0].tolist()) == {0, 1, 2}
    assert mix_chunks(wiki, squad, 0.0) is wiki
