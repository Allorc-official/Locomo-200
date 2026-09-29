# What the benchmark measures about your agent

LoCoMo-200 evaluates an agent as a complete system: it must ingest the history of a long
conversation, retain durable facts across sessions, and then answer questions accurately,
completely, and without inventing anything — through the same interface a user would use.

## Question categories

### Category 1 — multi-hop (100 questions)
Requires facts from **two or more sessions** (and often from both speakers) to answer.
This is the core memory test: a system that remembers only recent context, or that
retrieves a single relevant node, will produce partial answers.

*Example shape:* "What did X and Y plan to do together?" where one plan is stated in
session 6 and the other in session 14.

### Category 2 — temporal (70 questions)
Dates, ordering, durations, and event state over time: when something happened, what came
first, whether an event was planned or completed, how long after another event it occurred.

*Example shape:* "When did X go camping?" — the answer may be relative ("the week before
27 June") and must be resolved against the source session date at the precision the source
supports.

### Category 4 — single-hop (10 questions)
Direct recall of one clearly stated fact. These should be near-perfect for any working
memory system; they act as a sanity floor.

### Category 5 — adversarial (20 questions)
The premise is **not** present in the conversation (for example, asking about a family
event nobody planned). The correct behavior is abstention: state that the records do not
establish the premise, without substituting a nearby fact. These test groundedness and
resistance to hallucination.

## Difficulty metadata

Every question carries evidence statistics that the judge report breaks down by:

- `evidence_session_count` — how many sessions the answer spans;
- `evidence_turn_count` — how many evidence turns are involved;
- `evidence_span` — distance between the first and last evidence turn;
- `oldest_evidence_age` / `newest_evidence_age` — how many sessions back the evidence lies;
- `query_depth` — the question's position in the conversation.

A system that fits only a recent window will score well on low-age items and poorly on
high-age items; a system that over-retrieves will often lose on adversarial items by
answering with unsupported material.

## What a good score means

| Score band | Reading |
|---|---|
| > 85 | Strong memory: multi-session facts are recalled completely, temporal answers keep source precision, adversarial items are refused. |
| 70–85 | Solid recall with occasional omissions (typically missing one of several items) or unsupported extras. |
| 50–70 | Partial memory: frequent omissions on multi-hop questions, or answers that describe retrieval instead of facts. |
| < 50 | The system is not retaining conversation facts end-to-end. |

Because scoring is strict (any missing, wrong, or unsupported critical fact fails a
question), the useful diagnostic is **claim coverage** and the per-category breakdown —
not the headline number alone.

## What it does not measure

- Retrieval mechanics (index type, embeddings, query strategy) — only outcomes.
- Image understanding (text-only payload).
- World knowledge (category 3 was removed).
- Style. The judge ignores harmless, supported elaboration.
