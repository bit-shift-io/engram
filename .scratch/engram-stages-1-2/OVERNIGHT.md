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
