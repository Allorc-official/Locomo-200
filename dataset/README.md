# Dataset card — LoCoMo-200 (filtered subset)

## What this is

200 questions over 10 long-term conversations, filtered from the LoCoMo benchmark
(Maharana et al., ACL 2024) to questions whose answers are verifiable from the
conversation text and whose evidence references resolve. The conversations are relayed to
the agent as ordered, text-only session transcripts, one session at a time; questions are
asked afterwards.

## Files

| File | Description |
|---|---|
| `locomo200.json` | The filtered benchmark (10 samples × 20 questions) |
| `locomo200.manifest.json` | Selection manifest: quotas, exclusions, category mapping, hashes |

## Provenance and integrity

| | |
|---|---|
| Origin | [LoCoMo](https://github.com/snap-research/locomo) — Maharana et al., ACL 2024 |
| Upstream repository | `https://github.com/snap-research/locomo` — `data/locomo10.json` |
| Upstream SHA-256 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| This dataset SHA-256 | `411ec27105f3c7b632b9b509f9e25149c7c937cf971c053fb3c7e863a6627a5c` |
| Canonical (sorted) SHA-256 | `eed02816c3ca90eef1a1126c5db33fa45fd7578065e2b473b721e025f5d06e1e` |
| Upstream license | CC BY-NC 4.0 — attribution and modification notices in `../NOTICE.md` |

This dataset is an **independent, filtered derivative** of LoCoMo: it is not affiliated
with or endorsed by the LoCoMo authors, and the original dataset remains at the upstream
repository.

## Evaluation totals

| | Count |
|---|---|
| Conversations | 10 |
| History sessions replayed (one session per replay event) | 272 |
| Conversation turns | 5,782 |
| Questions asked (one fresh question session each) | 200 |
| Evaluation steps | **272 + 200 = 472** |

## Per-conversation statistics

| Conversation | Sessions | Turns | Questions (multi-hop / temporal / single-hop / adversarial) |
|---|---|---|---|
| conv-26 | 19 | 419 | 10 / 7 / 1 / 2 |
| conv-30 | 19 | 369 | 10 / 7 / 1 / 2 |
| conv-41 | 32 | 663 | 10 / 7 / 1 / 2 |
| conv-42 | 29 | 629 | 10 / 7 / 1 / 2 |
| conv-43 | 29 | 680 | 10 / 7 / 1 / 2 |
| conv-44 | 28 | 675 | 10 / 7 / 1 / 2 |
| conv-47 | 31 | 689 | 10 / 7 / 1 / 2 |
| conv-48 | 30 | 681 | 10 / 7 / 1 / 2 |
| conv-49 | 25 | 509 | 10 / 7 / 1 / 2 |
| conv-50 | 30 | 568 | 10 / 7 / 1 / 2 |
| **Total** | **272** | **5,782** | **100 / 70 / 10 / 20** |

## Record shape

Each sample:

```json
{
  "sample_id": "conv-26",
  "conversation": {
    "session_1": [{"speaker": "Caroline", "text": "...", "dia_id": "D1:1"}],
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "speaker_a": "...", "speaker_b": "..."
  },
  "qa": [
    {
      "question_id": "conv-26:qa:019",
      "source_qa_index": "19",
      "question": "What do Melanie's kids like?",
      "answer": "dinosaurs, nature",
      "category": "1",
      "category_name": "multi_hop",
      "answerability": "answerable",
      "evidence": ["D6:6", "D4:8"],
      "evidence_turns": [{"id": "D6:6", "session_number": 6, "text": "..."}],
      "difficulty": {"evidence_sessions": [4, 6], "evidence_session_count": 2, "evidence_turn_count": 2, "evidence_span": 2, "query_depth": 19, "oldest_evidence_age": 15, "newest_evidence_age": 13}
    }
  ]
}
```

Notes:

- `answerability`: `answerable` (a supported answer exists) or `adversarial` (the premise
  is not in the conversation; the correct behavior is to say the records do not establish
  it — the reference `answer` is `null`).
- Image URLs and captions from upstream are retained in the raw conversation objects but
  are **not** sent to the agent by the runner (text-only payload).
- `difficulty` supports score breakdowns (sessions, turns, span, evidence age, query depth).
