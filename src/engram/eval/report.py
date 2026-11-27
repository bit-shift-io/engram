"""Write eval results and render markdown comparison tables."""

import json
from pathlib import Path


def write_run_report(run_dir, result: dict) -> Path:
    d = Path(run_dir) / "eval"
    d.mkdir(exist_ok=True)
    (d / "reconstruction.json").write_text(json.dumps(result, indent=2))
    (d / "reconstruction.md").write_text(markdown_table([result]) + "\n")
    return d / "reconstruction.json"


def markdown_table(results: list[dict]) -> str:
    rows = ["| K | compression | real acc | zeroed acc | shuffled acc | real ROUGE-L | zeroed ROUGE-L | shuffled ROUGE-L |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["k"]):
        acc = [f"{r[c]['token_accuracy']:.3f}" for c in ("real", "zeroed", "shuffled")]
        rl = [f"{r[c]['rouge_l']:.3f}" for c in ("real", "zeroed", "shuffled")]
        rows.append(f"| {r['k']} | {r['compression_ratio']:g}x | " + " | ".join(acc + rl) + " |")
    return "\n".join(rows)


def load_results(run_dirs) -> list[dict]:
    return [json.loads((Path(d) / "eval" / "reconstruction.json").read_text()) for d in run_dirs]


def load_control_result(path) -> dict | None:
    """Cached text-QA control result ({n, em, f1}), or None if it has not been run."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# Stage 2 accepted run (runs/20261125-163535): name -> (baseline, lowest ok, highest ok)
STAGE2_GATES = {
    "full_gap": (1.0, 0.9, None),  # zeroed - real <full> val loss, nats
    "terse_article_ratio": (0.10, 0.05, 0.15),
    "full_article_ratio": (0.91, 0.80, 1.05),
    "terse_rouge_l": (0.184, 0.164, None),
}
# SQuAD dev <full> reconstruction (engram.eval.recall), checked only when the row has them.
# Baseline = K=16 Stage 2 (runs/20261125-163535/eval/recall.json, n=40); bands are unreviewed:
# a joint stage that trains <full> on SQuAD paragraphs should beat the untrained baseline.
SQUAD_GATES = {
    # generated / reference <full> length; floor catches collapse (Stage 3 gave ~0.02)
    # without flagging K=16 Stage 2's healthy 0.50
    "squad_full_length_ratio": (0.50, 0.40, 1.30),
    # gold answer appears in the <full> output; floor allows a ~5-point drop below
    # the 0.25 recall gate the joint stage only runs after passing
    "answer_recall": (0.075, 0.20, None),
}
K32_THRESHOLD = 0.5  # memory F1 / text-QA control F1 below this -> run a K=32 arm


def stage3_gates(row: dict, control: dict | None) -> dict:
    """row: a Stage 3 validation row (full_gap, density metrics, qa_real_f1). Flags metrics outside
    the Stage 2 band and recommends K=32 when memory F1 / control F1 < 0.5 (None: no control)."""
    regressions = []
    for name, (_, lo, hi) in {**STAGE2_GATES, **SQUAD_GATES}.items():
        v = row.get(name)
        if v is None and name in SQUAD_GATES:
            continue
        if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
            regressions.append(name)
    ratio = None
    if control and control.get("f1") and row.get("qa_real_f1") is not None:
        ratio = round(row["qa_real_f1"] / control["f1"], 6)
    return {"metrics": {k: row.get(k) for k in [*STAGE2_GATES, *SQUAD_GATES, "qa_real_f1"]},
            "regressions": regressions, "memory_over_control_f1": ratio,
            "recommend_k32": None if ratio is None else ratio < K32_THRESHOLD}


def stage3_markdown(g: dict) -> str:
    rows = ["| metric | Stage 3 | Stage 2 | ok range | status |", "|---|---|---|---|---|"]
    for name, (base, lo, hi) in {**STAGE2_GATES, **SQUAD_GATES}.items():
        v = g["metrics"].get(name)
        rng = f"{lo if lo is not None else '-inf'} .. {hi if hi is not None else 'inf'}"
        status = "REGRESSION" if name in g["regressions"] else "ok"
        rows.append(f"| {name} | {'n/a' if v is None else f'{v:.3f}'} | {base} | {rng} | {status} |")
    r = g["memory_over_control_f1"]
    if r is None:
        rows.append("\nmemory ÷ control F1: n/a (text-QA control not run)")
    else:
        rec = "recommend a K=32 arm" if g["recommend_k32"] else "keep K=16 (no K=32 arm)"
        rows.append(f"\nmemory ÷ control F1: {r:.3f} (threshold {K32_THRESHOLD}) -> {rec}")
    return "\n".join(rows)


RECALL_GATE = 0.25  # Stage 3b: warm-up answer recall on 40 SQuAD dev paragraphs (DECISIONS Q2)


def recall_gate(recall: dict, threshold: float = RECALL_GATE) -> dict:
    """recall: engram.eval.recall result. Passed when answer recall >= threshold."""
    passed = recall["answer_recall"] >= threshold
    reason = (f"answer recall {recall['answer_recall']:.3f} >= {threshold}" if passed else
              f"answer recall {recall['answer_recall']:.3f} < {threshold}: the K=32 warm-up memory does "
              "not keep enough of the paragraph to answer from; stopping before the joint stage")
    return {**recall, "threshold": threshold, "passed": passed, "reason": reason}


def recall_gate_markdown(g: dict) -> str:
    return "\n".join([
        "# Stage 3b recall gate", "",
        f"- K={g['k']}, n={g['n']} SQuAD dev paragraphs",
        f"- answer recall: {g['answer_recall']:.3f} (threshold {g['threshold']})",
        f"- recon token F1: {g['recon_token_f1']:.3f}, length ratio: {g['length_ratio']:.3f}",
        f"- {'PASS' if g['passed'] else 'FAIL'}: {g['reason']}"])


def stage3b_final(qa: dict, recall: dict, density: dict, val: dict, control: dict | None) -> dict:
    """Final Stage 3b summary: memory QA real/shuffled/zeroed, control F1 and ratio, answer recall,
    regression gates (Stage 2 bands + SQuAD <full>) over the final evals and last val row."""
    row = {**val, **{k: density.get(k) for k in ("terse_article_ratio", "full_article_ratio", "terse_rouge_l")},
           "answer_recall": recall["answer_recall"], "squad_full_length_ratio": recall["length_ratio"],
           "qa_real_f1": qa["real"]["f1"]}
    g = stage3_gates(row, control)
    return {"qa": {c: qa[c] for c in ("real", "shuffled", "zeroed")}, "qa_n": qa["n"],
            "control_f1": control.get("f1") if control else None, "answer_recall": recall["answer_recall"],
            **g}


def stage3b_final_markdown(s: dict) -> str:
    rows = ["# Stage 3b final summary", "", "| memory | EM | F1 |", "|---|---|---|"]
    rows += [f"| {c} | {s['qa'][c]['em']:.3f} | {s['qa'][c]['f1']:.3f} |" for c in ("real", "shuffled", "zeroed")]
    ctrl = "n/a" if s["control_f1"] is None else f"{s['control_f1']:.3f}"
    rows += ["", f"text-QA control F1: {ctrl} (QA n={s['qa_n']})",
             f"answer recall: {s['answer_recall']:.3f}", "", stage3_markdown(s)]
    return "\n".join(rows)


# ---- fact-objective pipeline (scripts/fact_objective.py) ----------------------------------------

FACT_RECALL_PASS = 0.15
# Measured earlier; hardcoded. ROUGE-L tuples are (real, shuffled, zeroed).
FACT_BASELINES = {
    # runs/20261125-090655 (K=16 Stage 1); answer recall not measured for this run
    "K=16 Stage 1": {"answer_recall": None, "recon_token_f1": None,
                     "wikitext": (0.226, 0.164, 0.051), "squad": (0.141, 0.103, 0.037)},
    # runs/k32_wikionly/20261126-164326 (K=32 WikiText-only)
    "K=32 WikiText-only": {"answer_recall": 0.050, "recon_token_f1": 0.195,
                           "wikitext": (0.163, 0.136, 0.001), "squad": (0.109, 0.077, 0.000)},
}
FACT_STAGE2_RECALL = 0.075  # K=16 Stage 2 (runs/20261125-163535/eval/recall.json, n=40)


def fact_arm(k: int, run: str, ev: dict) -> dict:
    """ev: {recall: evaluate_recall result, wikitext/squad: evaluate_reconstruction results}.
    Pass = answer_recall >= 0.15 and real ROUGE-L > shuffled ROUGE-L on WikiText val."""
    rc = ev["recall"]["answer_recall"]
    real, shuf = ev["wikitext"]["real"]["rouge_l"], ev["wikitext"]["shuffled"]["rouge_l"]
    reasons = []
    if rc < FACT_RECALL_PASS:
        reasons.append(f"answer recall {rc:.3f} < {FACT_RECALL_PASS}")
    if not real > shuf:
        reasons.append(f"WikiText ROUGE-L real {real:.3f} <= shuffled {shuf:.3f}")
    rouge = {s: tuple(ev[s][c]["rouge_l"] for c in ("real", "shuffled", "zeroed")) for s in ("wikitext", "squad")}
    return {"k": k, "run": run, "answer_recall": rc, "recon_token_f1": ev["recall"]["recon_token_f1"],
            **rouge, "passed": not reasons, "reason": "; ".join(reasons)}


def fact_summary_markdown(arms: list[dict]) -> str:
    rows = ["# Fact objective summary", "",
            f"Pass: answer recall >= {FACT_RECALL_PASS} and WikiText real ROUGE-L > shuffled.", "",
            "| run | answer recall | recon F1 | WikiText ROUGE-L real/shuf/zero | SQuAD ROUGE-L real/shuf/zero | pass |",
            "|---|---|---|---|---|---|"]
    f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    t = lambda r: " / ".join(f"{v:.3f}" for v in r)  # noqa: E731
    for name, b in FACT_BASELINES.items():
        rows.append(f"| {name} (baseline) | {f(b['answer_recall'])} | {f(b['recon_token_f1'])} | "
                    f"{t(b['wikitext'])} | {t(b['squad'])} | - |")
    for a in arms:
        status = "PASS" if a["passed"] else f"FAIL ({a['reason']})"
        rows.append(f"| fact K={a['k']} ({a['run']}) | {f(a['answer_recall'])} | {f(a['recon_token_f1'])} | "
                    f"{t(a['wikitext'])} | {t(a['squad'])} | {status} |")
    rows += ["", f"K=16 Stage 2 answer recall (reference): {FACT_STAGE2_RECALL:.3f}. "
             "Baselines: runs/20261125-090655 (K=16 Stage 1), runs/k32_wikionly/20261126-164326 (K=32)."]
    return "\n".join(rows)


# ---- SQuAD cloze pipeline (scripts/squad_cloze.py) ----------------------------------------------

SQUAD_CLOZE_RECALL_PASS = 0.15
SQUAD_CLOZE_GAP_PASS = 1.0  # nats: zeroed answer loss - real answer loss on the cloze recall eval
BASELINE_RUN = "runs/fact_objective/k16/20261126-212058"
BASELINE_RECALL = 0.075  # runs/fact_objective/summary.json, K=16 arm


def baseline_row(base_dir: str = BASELINE_RUN, recall: float = BASELINE_RECALL) -> dict:
    """Fact K=16 ckpt-3000: answer recall and the cloze recall eval read from its run dir."""
    cl = json.loads((Path(base_dir) / "eval" / "cloze_recall.json").read_text())
    return {"run": base_dir, "answer_recall": recall, "cloze": cl}


def squad_cloze_summary(run: str, recall: dict, cloze: dict, early_stop: dict | None,
                        baseline: dict | None) -> dict:
    """Pass = answer recall >= 0.15 AND cloze real beats zeroed answer loss by >= 1 nat; each
    condition is reported separately."""
    rc = recall["answer_recall"]
    gap = cloze["zeroed"]["answer_loss"] - cloze["real"]["answer_loss"]
    conds = {"answer_recall": {"value": rc, "threshold": SQUAD_CLOZE_RECALL_PASS,
                               "passed": rc >= SQUAD_CLOZE_RECALL_PASS},
             "cloze_gap": {"value": gap, "threshold": SQUAD_CLOZE_GAP_PASS,
                           "passed": gap >= SQUAD_CLOZE_GAP_PASS}}
    return {"run": run, "answer_recall": rc, "cloze": {c: cloze[c] for c in ("real", "shuffled", "zeroed")},
            "cloze_n": cloze["n"], "conditions": conds, "passed": all(c["passed"] for c in conds.values()),
            "stopped_early": early_stop is not None, "stop_reason": early_stop["reason"] if early_stop else "",
            "baseline": baseline}


def squad_cloze_markdown(s: dict) -> str:
    c = s["conditions"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    rows = ["# SQuAD cloze summary", "",
            f"- answer recall >= {c['answer_recall']['threshold']}: {mark(c['answer_recall']['passed'])} "
            f"({c['answer_recall']['value']:.3f})",
            f"- cloze real beats zeroed answer loss by >= {c['cloze_gap']['threshold']} nat: "
            f"{mark(c['cloze_gap']['passed'])} ({c['cloze_gap']['value']:.2f})",
            f"- overall: {mark(s['passed'])}",
            f"- stop reason: {s['stop_reason'] or 'ran all steps'}", "",
            "| run | answer recall | cloze loss real/shuf/zero | cloze hits real/shuf/zero (of n) |",
            "|---|---|---|---|"]

    def row(name, rc, cl):
        loss = " / ".join(f"{cl[k]['answer_loss']:.2f}" for k in ("real", "shuffled", "zeroed"))
        hits = " / ".join(str(cl[k]["hits"]) for k in ("real", "shuffled", "zeroed"))
        return f"| {name} | {rc:.3f} | {loss} | {hits} of {cl['n']} |"
    b = s.get("baseline")
    if b:
        rows.append(row(f"fact K=16 ckpt-3000 baseline ({b['run']})", b["answer_recall"], b["cloze"]))
    rows.append(row(f"squad cloze ({s['run']})", s["answer_recall"], {**s["cloze"], "n": s["cloze_n"]}))
    return "\n".join(rows)
