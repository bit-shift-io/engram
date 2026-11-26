# Stage 3b final summary

| memory | EM | F1 |
|---|---|---|
| real | 0.000 | 0.000 |
| shuffled | 0.000 | 0.000 |
| zeroed | 0.000 | 0.005 |

text-QA control F1: 0.199 (QA n=40)
answer recall: 0.000

| metric | Stage 3 | Stage 2 | ok range | status |
|---|---|---|---|---|
| full_gap | -0.118 | 1.0 | 0.9 .. inf | REGRESSION |
| terse_article_ratio | 0.302 | 0.1 | 0.05 .. 0.15 | REGRESSION |
| full_article_ratio | 1.276 | 0.91 | 0.8 .. 1.05 | REGRESSION |
| terse_rouge_l | 0.020 | 0.184 | 0.164 .. inf | REGRESSION |
| squad_full_length_ratio | 1.455 | 0.5 | 0.7 .. 1.3 | REGRESSION |
| answer_recall | 0.000 | 0.075 | 0.075 .. inf | REGRESSION |

memory ÷ control F1: 0.000 (threshold 0.5) -> recommend a K=32 arm
