# Overnight log — 2026-11-24/25

## Stage 1 run #1 (runs/20261124-170521) — FAILED
Config: configs/stage1.yaml as built (lr 2e-3, 800 steps, decoder LoRA on). Val loss plateau 2.84.
Eval (n=32): real acc 0.017, zeroed 0.013, shuffled 0.015 → memory ignored.

## Diagnosis (debug agent)
- Not a code bug. Memory collapses to a constant soft prompt (cross-chunk cosine 0.9935).
- SmolLM2 massive activations (dims 17, 507) swamp chunk-specific signal at query slots.
- Decoder LoRA shortcut: learns LM / memorises chunks without reading memory.
- Frozen decoder can read memory when overfitting few chunks (4 chunks: real 0.48 vs shuffled 0.30).

## Short experiments (150 steps × 4 chunks, 400 chunks) — val real vs shuffled loss
| Run | Setup | real | shuffled | gap? |
|---|---|---|---|---|
| A | baseline | 2.636 | 2.666 | tiny |
| B | frozen decoder | 2.968 | 2.964 | no |
| C/D | centred memory (±frozen dec) | 2.70/3.17 | 2.70/3.16 | no |
| F | LoRA r16 + MLP + enc LR×5 | 2.763 | 2.763 | no |
| G/H | decoder input dropout | 3.24/2.84 | 3.22/2.88 | no |
| I/J/K | 64 chunks repeated | — | — | no on val |
| L | segment-mean residual, dec LoRA | 3.080 | 3.085 | no |

## Pending / next
- M (segment residual, frozen decoder), then outlier-dim masking, mid-layer memory, pooled hidden-state residual, K=64.
- Then full Stage 1 run with the best setting (or decoder_lora: false), eval, Stage 2 if a gap opens.

## Round 2 short experiments (frozen decoder unless noted) — val real / zeroed / shuffled loss
Note: original "L" silently used frozen decoder (repo default decoder_lora: false); L2 is the true decoder-LoRA run. Guard now asserts the setting.
| Run | Setup | real | zeroed | shuffled |
|---|---|---|---|---|
| M | segment-mean residual | 3.080 | 3.859 | 3.085 |
| L2 | segment residual + decoder LoRA | 2.752 | 3.000 | 2.760 |
| N | massive dims 17/507 masked | 2.923 | 3.866 | 2.929 |
| O | layer 15 + dims masked | 2.832 | 3.792 | 2.838 |
| P | pooled hidden-state residual, layer 15, masked | 2.899 | 3.778 | 2.898 |
| Q | P with K=64 (2x compression) | 2.961 | 4.295 | 2.964 |
Best real-vs-shuffled gap ≈ 0.008 nats (target ≥ 0.05). Even K=64 doesn't open a gap in 150 steps.
Conclusion: the bottleneck is the frozen decoder learning to read soft-prompt memory at all; the only positive signal so far is heavy repetition of a few chunks.
New config options (all default off): decoder_lora, lora_r/alpha/targets, center_memory, decoder_input_dropout, memory_segment_residual, memory_layer, memory_mask_dims, memory_pooled_residual. 32 fast tests pass.

## Stage 1 run #2 — started ~22:40, configs/stage1.yaml + decoder_lora: false
Result (runs/20261124-214047): val loss 2.737 (run #1: 2.842). Eval n=32: acc real 0.015 / zeroed 0.000 / shuffled 0.012; ROUGE-L 0.168 / 0.018 / 0.153. Still ~no real-vs-shuffled gap → FAILED.

## Curriculum — started ~23:26
A: 64 fixed train chunks, 400 steps (~50 epochs), frozen decoder. B: init from A/ckpt-400, full stage1 config (6400 chunks, 800 steps). Eval after each.
Result A (runs/20261124-232513): final train loss 2.92 on 64 chunks after ~50 epochs — did NOT memorise them. Val 3.32. Eval acc real 0.011 / zeroed 0.015 / shuffled 0.012.
Result B (runs/20261125-002245): val loss 2.655 (best so far). Eval acc real 0.017 / zeroed 0.000 / shuffled 0.012; ROUGE-L 0.183 / 0.000 / 0.149. Small gap, still FAILED vs ≥0.80 target.

## Capacity test C — started ~02:08
16 chunks, 300 steps (~150 epochs), encoder LoRA r64/alpha128 on attention + MLP, frozen decoder. Question: can memory carry chunk info at all at this setup?
Result C (runs/20261125-020719): train loss ~4.0 → ~3.8, flat/noisy, ABOVE val (3.56). 16 chunks × 150 epochs not memorised at all, though the debug agent's own script memorised 4 chunks with a frozen decoder → suspect a bug in the scripts/train.py → trainer path. Debug agent investigating (~02:45).
Debug finding (~03:05): NO trainer bug — trainer path matches a hand loop exactly (losses 4.45/4.097/3.854 at steps 1/6/11 on both). Flat curves explained by: frozen decoder reads memory very slowly; C's r64 LoRA too aggressive (step-1 grad norm 95 vs clip 1.0); first 16 train chunks are harder text (plain-LM loss 3.92 vs 3.28 on val), so train > val is expected. Correction: one.py never memorised 4 chunks (loss 2.51, acc 0.48 vs 0.30 shuffled). C2 (16 chunks, r8, 200 steps): train loss 4.22 → 3.84, ~0.1 below plain LM. New regression test test_frozen_decoder_trainer_step_updates_encoder_side_only; 33 fast tests pass.

## Long test D — started ~03:07
16 chunks, 1200 steps × batch 4 (~300 epochs), default r8, frozen decoder. Does the decoder ever learn to read memory with enough repetition? (Plain-LM loss on these chunks: 3.92.)
Result D (runs/20261125-030545): train loss 4.05 → 2.73 (plain LM 3.92 on these chunks); val rose 3.40 → 3.96 (overfit).
Teacher-forced on the 16 train chunks: real 2.701 (acc 0.486) / shuffled 4.145 (0.323) / zeroed 4.865 (0.255).
→ FIRST CLEAR POSITIVE: a 1.45-nat real-vs-shuffled gap. The frozen decoder DOES learn to read memory given enough repetition. It just doesn't generalise yet — a data/compute problem, not a bug.

## Middle test E — started ~04:20
512 chunks, 2200 steps × batch 4 (~17 epochs), frozen decoder. Does reading memory transfer to val chunks?
Result E (runs/20261125-041111): train loss 3.60 → 2.37; trainer val (64 chunks) best 2.88 @ step 800, 3.00 at end.
Teacher-forced, 16 chunks each:
| set | real | shuffled | zeroed |
|---|---|---|---|
| train | 2.656 (acc 0.470) | 3.723 (0.356) | 4.867 (0.253) |
| val (unseen) | 2.698 (acc 0.471) | 2.976 (0.436) | 3.940 (0.336) |
→ FIRST GENERALISATION: a 0.28-nat real-vs-shuffled gap on UNSEEN chunks (every earlier run: ≤0.01). Still far from the 80% greedy-reconstruction target.

## Not run
Stage 2 — Stage 1 hasn't passed, so training <terse> on top would test nothing.

## Option 2 — memory-position K/V adapter (2026-11-25)
512 chunks, 500 steps x bs4, lr 2e-3, frozen decoder, seed 0. Held-out teacher-forced, 16 chunks.
| Run | real / shuffled / zeroed loss | acc real / shuffled |
|---|---|---|
| (a) plain frozen | 2.841 / 2.839 / 3.788 | 0.465 / 0.465 |
| (b) adapter r8 | 2.581 / 2.699 / 3.303 | 0.473 / 0.468 |
| (c) adapter r32 | 2.655 / 2.652 / 3.571 | 0.479 / 0.479 |
r8 gives a 0.118-nat held-out gap (single seed); r32 none. stage1.yaml now: 2000 chunks, 3000 steps, adapter rank 8.

## Option 1 — overnight Stage 1, runs/20261125-090655 (2026-11-25)
stage1.yaml: 2000 chunks, 3000 steps x bs4, frozen decoder + mem adapter r8. ~2.6 h. Held-out, 16 chunks, teacher-forced.
| Step | real / shuffled / zeroed loss | acc real / shuffled | loss gap | acc gap |
|---|---|---|---|---|
| 500 | 2.277 / 2.681 / 3.998 | 0.498 / 0.481 | 0.40 | +1.7 |
| 1000 | 2.164 / 2.887 / 3.979 | 0.515 / 0.469 | 0.72 | +4.6 |
| 1500 | 2.111 / 3.176 / 3.915 | 0.506 / 0.442 | 1.07 | +6.4 |
| 2000 | 2.054 / 3.254 / 3.893 | 0.518 / 0.441 | 1.20 | +7.7 |
| 2500 | 2.043 / 3.333 / 3.896 | 0.514 / 0.442 | 1.29 | +7.2 |
| 3000 | 2.047 / 3.362 / 3.890 | 0.516 / 0.442 | 1.32 | +7.4 |
Loss target (>=0.5) met; accuracy target (+10) not met — accuracy plateaued from step 2000. val_loss flat at 2.25 from step 2100.

## Stage 2 — runs/20261125-123127 (2026-11-25)
init from runs/20261125-090655/ckpt-3000.pt; decoder LoRA lr 3e-4, encoder side + adapter r8 ~1e-4; 500 steps, ~2.7 h.
<full> held-out real / shuffled loss: s100 2.041/3.081 (gap 1.04) · s200 1.998/3.196 · s300 1.991/3.198 · s400 1.985/3.165 · s500 1.984/3.166 (gap 1.18, acc 0.531 vs 0.462). Gap target >=0.4 met.
terse_val_loss 3.08 -> 2.88. Density (32 XSum): full len 25.7, terse len 26.5, ratio 4.37 (target <=0.25 FAILS),
terse ROUGE-L 0.195 vs base TL;DR 0.108. Ratio fails because <full> stops after ~26 tokens on 4-chunk (64-vector)
memory — <full> was only ever trained on 1-chunk (16-vector) memory, so the comparison length is broken, not terse.

## Stage 2 + multi-chunk <full> — runs/20261125-142812 (2026-11-25)
init from runs/20261125-123127/ckpt-500.pt, multi_full_ratio 0.5 (25% 1-chunk full / 25% article full / 50% terse), 400 steps, 15.5 s/step.
<full> held-out real/shuffled: s100 2.003/3.160 · s400 1.989/3.163 (gap 1.17, acc 0.527 vs 0.454). multi_val_loss 2.75 -> 2.66; terse_val 2.86 -> 2.82.
Density: full len 34.7, terse 25.8, ratio 1.09 (target <=0.25 FAILS); terse ROUGE-L 0.192 vs base 0.108.
Inspection: on XSum articles <full> writes a one-sentence news-style summary and stops (13-34 tokens) — it follows the
input's genre (articles always meant summary for 500 steps) over the <full> control token. On WikiText chunks <full>
writes the full 128 tokens, on-topic but paraphrased/repetitive: memory holds gist, not verbatim text.

## Stage 2 + control vectors — runs/20261125-163535 (2026-11-25)
init from Stage 1 ckpt-3000; control_vectors on; mix 20% 1-chunk full / 40% article full / 40% terse; 600 steps, 19 s/step.
<full> held-out real/shuffled: s100 2.147/2.972 · s300 2.039/3.130 · s600 2.017/3.117 (gap 1.10, acc 0.521 vs 0.449). Gap target met.
val s600: full 2.239, multi (article full) 2.530 (prev run 2.657), terse 2.849.
Density (32 XSum): article 321.8, full 266.1 (was 34.7), terse 22.4. terse/full 0.297 (target <=0.25: close miss),
terse/article 0.100, full/article 0.908. terse ROUGE-L 0.184 vs base 0.108. Control vectors fixed <full> stopping early.
