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
