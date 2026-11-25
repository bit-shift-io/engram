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
K32_THRESHOLD = 0.5  # memory F1 / text-QA control F1 below this -> run a K=32 arm


def stage3_gates(row: dict, control: dict | None) -> dict:
    """row: a Stage 3 validation row (full_gap, density metrics, qa_real_f1). Flags metrics outside
    the Stage 2 band and recommends K=32 when memory F1 / control F1 < 0.5 (None: no control)."""
    regressions = []
    for name, (_, lo, hi) in STAGE2_GATES.items():
        v = row.get(name)
        if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
            regressions.append(name)
    ratio = None
    if control and control.get("f1") and row.get("qa_real_f1") is not None:
        ratio = round(row["qa_real_f1"] / control["f1"], 6)
    return {"metrics": {k: row.get(k) for k in [*STAGE2_GATES, "qa_real_f1"]},
            "regressions": regressions, "memory_over_control_f1": ratio,
            "recommend_k32": None if ratio is None else ratio < K32_THRESHOLD}


def stage3_markdown(g: dict) -> str:
    rows = ["| metric | Stage 3 | Stage 2 | ok range | status |", "|---|---|---|---|---|"]
    for name, (base, lo, hi) in STAGE2_GATES.items():
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
