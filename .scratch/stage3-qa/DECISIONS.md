# Decisions: Stage 3 QA

### Q1: QA data source
**Decision:** SQuAD v1.1 (train for training, dev for held-out evaluation).
- **Why:** Free, extractive, standard EM/F1, ~87k questions over ~19k distinct paragraphs.
- **Implication:** One download with network; no API cost.
- **Alternatives considered:** Synthetic QA from a larger LLM over WikiText/XSum — deferred until SQuAD shows the concept works.

### Q2: Memory capacity
**Decision:** Keep K=16. Run a K=32 arm only if memory F1 ÷ text-QA control F1 < 0.5.
- **Why:** Slow CPU; K=32 doubles decoder memory length for every later stage. Smaller T0 units may suit T1 slots.
- **Implication:** A low ratio is ambiguous (capacity vs. training) until the fallback runs.
- **Alternatives considered:** Early K=16 vs K=32 comparison — rejected for cost.

### Q3: Training mix
**Decision:** 50% `<answer>`, 25% `<full>` replay, 25% `<terse>` replay, balanced from step 0.
- **Why:** A new task overrides old control behaviour unless the mix is balanced from the start.
- **Implication:** Stage 2 gates checked at every validation.
- **Alternatives considered:** 70/15/15 — more QA signal, higher regression risk.

### Q4: Full-text baseline
**Decision:** The text-QA control is the denominator: same frozen base plus a same-size LoRA, trained on SQuAD with the paragraph as plain text, same steps. Zero-shot base model also reported.
- **Why:** Zero-shot 135M is weak on QA; matching it proves little. The control isolates memory vs. text.
- **Implication:** One extra training run (a few hours).
- **Alternatives considered:** Zero-shot only — rejected as too weak a bar.

### Q5: `<answer>` control
**Decision:** New control token `<answer>` with its own flag vector, initialised to zero. Question tokens follow it; loss on answer tokens only.
- **Why:** Matches `<full>`/`<terse>`; zero init leaves Stage 2 behaviour unchanged at step 0.
- **Alternatives considered:** Question as plain prefix without a task token — rejected; prefix tokens get overridden by input type.

### Q6: Scoring
**Decision:** Greedy decode, max ~32 new tokens, standard SQuAD normalisation (lowercase, strip punctuation and articles); EM and token F1 against all gold answers, take the max.

## Assumptions
- Most SQuAD paragraphs fit in ≤ 4 chunks.
- The QA objective may push the encoder to keep facts within 16 vectors.

## Trade-offs
- Extractive QA rewards copying surface spans, not abstract meaning. Accepted for a first test.

## CONTEXT.md additions
- **Answer task** — `<answer>` control: memory + question → short answer.
- **Text-QA control** — baseline model reading the paragraph as plain text instead of memory, trained identically.
