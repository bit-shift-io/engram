# Fact objective summary

Pass: answer recall >= 0.15 and WikiText real ROUGE-L > shuffled.

| run | answer recall | recon F1 | WikiText ROUGE-L real/shuf/zero | SQuAD ROUGE-L real/shuf/zero | pass |
|---|---|---|---|---|---|
| K=16 Stage 1 (baseline) | n/a | n/a | 0.226 / 0.164 / 0.051 | 0.141 / 0.103 / 0.037 | - |
| K=32 WikiText-only (baseline) | 0.050 | 0.195 | 0.163 / 0.136 / 0.001 | 0.109 / 0.077 / 0.000 | - |
| fact K=16 (runs_smoke_fact/k16/20261126-210004) | 0.000 | 0.000 | 0.019 / 0.023 / 0.008 | 0.008 / 0.016 / 0.008 | FAIL (answer recall 0.000 < 0.15; WikiText ROUGE-L real 0.019 <= shuffled 0.023) |
| fact K=32 (runs_smoke_fact/k32/20261126-210514) | 0.000 | 0.018 | 0.012 / 0.008 / 0.074 | 0.047 / 0.019 / 0.062 | FAIL (answer recall 0.000 < 0.15) |

K=16 Stage 2 answer recall (reference): 0.075. Baselines: runs/20261125-090655 (K=16 Stage 1), runs/k32_wikionly/20261126-164326 (K=32).
