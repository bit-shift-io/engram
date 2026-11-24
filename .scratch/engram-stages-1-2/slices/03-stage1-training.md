Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# Stage 1 reconstruction training on WikiText

## What to build
- The WikiText-103 data pipeline: clean, tokenize, split into 128-token chunks, with train/val splits and a configurable sample count.
- A training loop with grad accumulation, a cosine LR schedule, periodic validation loss, checkpoints (trainable params only) and resume.
- `scripts/train.py --config configs/stage1.yaml` runs end to end and writes `runs/<timestamp>/` (config snapshot, `metrics.jsonl`, checkpoints).

## Files to create/modify
- src/engram/data/chunking.py
- src/engram/data/reconstruction.py
- src/engram/train/losses.py
- src/engram/train/trainer.py
- scripts/train.py
- configs/stage1.yaml, configs/smoke.yaml
- tests/test_chunking.py, tests/test_trainer.py

## Test approach
- Chunking: exact chunk lengths; the short tail is dropped or padded consistently.
- Trainer: a smoke config (tiny sample, 5 steps) completes, writes metrics and a checkpoint, and resumes from it.
- Manual: run `configs/stage1.yaml` and watch val loss fall.

## Acceptance criteria
- [ ] The smoke run finishes in under 2 minutes.
- [ ] The Stage 1 default run finishes in under ~4 hours on the Intel Mac.
- [ ] Val loss drops steadily, and checkpoints load back into `Engram`.

## Blocked by
02

## Gotchas
- Set `torch.set_num_threads` to the physical core count (6). Hyperthreads slow things down.
- Cache the tokenized dataset to disk. Re-tokenizing each run wastes minutes.
