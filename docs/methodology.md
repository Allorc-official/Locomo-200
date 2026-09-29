# Methodology — how LoCoMo-200 was built

## Source

LoCoMo (Maharana et al., ACL 2024): 10 very long-term two-speaker conversations
(19–32 sessions each; 5,782 turns in this subset), with question–answer annotations and
evidence turn references. Upstream file `data/locomo10.json`, SHA-256
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

## Why we filtered

The upstream question set mixes capabilities (including open-domain world knowledge),
contains questions whose evidence does not resolve, and includes items whose gold answer
is not verifiable from the conversation text (for example, facts that exist only in shared
images). Mixing those into a memory benchmark makes the score depend on annotation quality
rather than on the agent. The filtered subset is designed so that **every retained question
has a text-verifiable answer and a resolvable evidence path**.

## Filtering pipeline

Starting from all 1,986 upstream questions (across 5 categories):

| Step | Rule | Excluded |
|---|---|---|
| 1 | Category 3 (open-domain) removed — not a memory question | 96 |
| 2 | Category 1 must have cross-session evidence (multi-hop means ≥2 sessions) | 12 |
| 3 | Category 5 must have a null reference answer (a true false-premise item) | 2 |
| 4 | Evidence references must resolve to turns in the conversation | 6 |
| 5 | Quota: 20 questions per conversation, balanced per category | 1,670 |

Final selection: **200 questions** — per conversation 10 multi-hop, 7 temporal,
2 adversarial, 1 single-hop (100 / 70 / 20 / 10 overall). Within each category and
conversation, questions are sampled at evenly spaced ranks along evidence difficulty
(evidence-session count, turn count, span), with a stable source-index tie-break. Full
details and the per-question decisions are in `dataset/locomo200.manifest.json`
(`exclusions`, `selection`, `per_conversation_counts_by_category`).

## Payload boundary and replay conditions

The runner sends the agent:

1. each session in order — `speaker` and `text` only (no image URLs, captions, evidence,
   categories, or reference answers). **Each replay event carries exactly one session**,
   sent once, in chronological order;
2. then each question's text, **one fresh question session per question**, only after all
   of that conversation's sessions have been replayed.

Totals: **10 conversations · 272 history-session replays · 5,782 turns · 200 questions
(100 multi-hop, 70 temporal, 10 single-hop, 20 adversarial) = 472 evaluation steps.**
At question time the agent never sees the original transcripts again; only its own memory
can carry the facts.

The agent never sees the reference answer, the evidence ids, or any difficulty metadata.
This matches the "black-box agent" setup: the benchmark measures the agent's memory and
answering as a system, not a particular retrieval implementation.

## Known limitations

- **Text-only.** Facts that exist only in shared images cannot be captured by the agent;
  such questions were filtered out where they were detectable, but some single facts in
  retained questions may still depend on image content. If a question's critical claim is
  absent from the conversation text, no answer can satisfy the strict judge — treat such
  items as dataset-bounded.
- **Annotation precision.** Some gold answers assert a tighter precision (e.g., "end of
  October") than the source text states. The judge checks claims against the conversation,
  so such items are effectively unpassable; they are a small minority.
- **Language coverage.** Source conversations are English and casual in register.
- **Non-commercial use only** — see `../NOTICE.md`.
