# Decisions — fact-qa-stages

### Q1: What is the goal?
**Decision:** Prove two things: answers from memory, and different output lengths from the same memory.
- **Why:** `<answer>`, `<full>` and `<terse>` are demonstrations, not products.
- **Implication:** `<full>` need not write out every fact. The earlier "decoder fix for `<full>`" is dropped.
- **Alternatives considered:** Train `<full>` to express facts — rejected; answer recall in `<full>` is not a goal.

### Q2: Where does QA training start?
**Decision:** From the SQuAD cloze checkpoint, not the Stage 2 lineage.
- **Why:** It stores 3× more blanked facts (cloze recall 0.168 vs 0.056); Stage 3 on older memory reached 0.24× control F1.
- **Implication:** Length control must be rebuilt: this lineage never trained `<terse>`, control vectors or multi-chunk `<full>`.

### Q3: One stage or two?
**Decision:** Two: Stage A density, then Stage B answer, with a gate between.
- **Why:** A failure points at one stage; each reuses an existing recipe (Stage 2, Stage 3).
- **Alternatives considered:** One combined stage — rejected; four objectives at once on a decoder new to control vectors, and modes have overridden each other before.

### Q4: Gate thresholds
**Decision:** Fact gate: cloze recall ≥ 0.13 (~80% of 0.168), cloze loss gap ≥ 1.5 nats (now 1.93). Final: QA F1 ≥ 0.5× text-QA control.
- **Why:** Facts outrank length control (user).
- **Implication:** Lost facts stop the pipeline; missed length bands only warn.

## Assumptions
- Control vectors and a fresh decoder LoRA can be added to a checkpoint that lacks them (zero-init, no-op until trained).
- Stage 2 / Stage 3 data mixes and LRs carry over with cloze replay added.

## Trade-offs
- Two runs (~7–8 h CPU) over one, for attributable failures.

## CONTEXT.md entries added
- **Fact gate**, **Density stage** — see `docs/CONTEXT.md`.
