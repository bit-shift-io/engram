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
