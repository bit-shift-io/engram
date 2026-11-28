from engram.eval.report import fact_qa_final, fact_qa_final_markdown

QA = {"n": 200, "real": {"em": 0.3, "f1": 0.4}, "shuffled": {"em": 0.1, "f1": 0.15}, "zeroed": {"em": 0.0, "f1": 0.05}}
CLOZE = {"cloze_recall": 0.2}
GOOD_VAL = {"full_gap": 1.5, "terse_rouge_l": 0.2}
DENSITY = {"terse_article_ratio": 0.10, "full_article_ratio": 0.9, "terse_rouge_l": 0.2}
CONTROL = {"n": 200, "em": 0.6, "f1": 0.69}


def _final(qa=QA, control=CONTROL, density=DENSITY):
    return fact_qa_final(qa, CLOZE, density, GOOD_VAL, control)


def test_ratio_at_or_above_half_passes():
    s = _final()
    assert s["ratio"] == round(0.4 / 0.69, 6) and s["ratio_passed"] is True


def test_ratio_below_half_fails():
    s = _final(qa={**QA, "real": {"em": 0.1, "f1": 0.2}})
    assert s["ratio_passed"] is False


def test_missing_control_is_undefined_not_pass():
    s = _final(control=None)
    assert s["ratio"] is None and s["ratio_passed"] is None
    assert "undefined" in fact_qa_final_markdown(s)


def test_regression_misses_are_warnings_beside_qa_result():
    s = _final(density={**DENSITY, "terse_article_ratio": 0.5})
    assert "terse_article_ratio" in s["warnings"]
    md = fact_qa_final_markdown(s)
    assert "WARNING" in md and "terse_article_ratio" in md


def test_markdown_shows_all_memory_conditions_and_cloze_recall():
    md = fact_qa_final_markdown(_final())
    for c in ("real", "shuffled", "zeroed"):
        assert f"| {c} |" in md
    assert "0.050" in md and "cloze recall: 0.200" in md
