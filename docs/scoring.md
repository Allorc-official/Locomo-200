# Scoring

The judge scores each answer semantically against the reference answer **and the
conversation history**. It is strict on facts and generous on style.

## Strict binary score (primary)

A question scores **1** only when every required fact is present in the answer and
supported by the conversation; otherwise **0**. Concretely, 0 is given for:

- a required fact missing (partial answers fail, not partially score);
- a wrong or contradicted fact;
- material unsupported additions (invented events, rankings, or precision);
- presenting a plan, interest, or suggestion as a completed action (or vice versa);
- failing to abstain on an adversarial premise by substituting an unrelated fact.

Harmless, supported elaboration (extra correct details, natural wording) does not fail a
question. No answer is compared by exact string.

## Diagnostics

- **Claim coverage** — the fraction of required claims entailed by the answer. Useful for
  measuring near-misses: 0.83 means "one required item out of six missing".
- **Token F1** — lexical overlap between the answer and the reference answer (quick,
  deterministic, insensitive to inference).
- **Flags** — per question: temporal error, unsupported material claim, contradiction,
  adversarial hallucination.

## Aggregates

The judge report includes the strict score out of 100 and the diagnostics overall and by:

- category (multi-hop / temporal / single-hop / adversarial),
- evidence-session count and evidence-turn count (1, 2, 3, 4+),
- evidence span, newest-evidence age, query depth.

## Reproducibility and variance

- LLM judging is not deterministic on borderline answers; expect roughly ±1 question
  deviation per conversation. **Run the judge at least three times and average** for
  published numbers.
- The judge is resumable: rerunning with the same `--output` file continues where it
  stopped (`--resume` when the output exists).
- Record the judge model and version. The reference implementation uses a strict binary
  rubric (judge version `locomo-strict-binary-v6` in `judge/judge.py`) and an
  OpenAI-compatible chat completions endpoint.

## Interpreting failures

| Signal | Meaning |
|---|---|
| Low claim coverage, high token F1 | The answer is wordy but misses required specifics — an enumeration problem. |
| High `unsupported_material_claim` rate | The system adds inferred facts, rankings, or precision — a grounding problem. |
| High `adversarial_hallucination` rate | False-premise questions are answered instead of refused. |
| `temporal_error` | Dates/order/plan-vs-outcome are mishandled. |
| Claim absent from the conversation text | Dataset-bounded item (see `methodology.md` limitations); no answer can pass. |
