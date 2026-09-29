# LoCoMo-200 — Long-Term Conversational Memory Benchmark

A filtered, text-verifiable 200-question subset of [LoCoMo](https://github.com/snap-research/locomo)
(Maharana et al., ACL 2024) for evaluating **long-horizon conversational memory of AI
agents end-to-end**: an agent receives the sessions of a long conversation in order, then
must answer questions about people, events, times, and quantities that were mentioned
across those sessions — with nothing but its own memory to rely on.

This package is self-contained: dataset, a runner that drives any agent through a simple
HTTP contract, and a strict semantic judge. Bring your own agent.

---

## Why this benchmark

Most memory evaluations test a retrieval index, a fixed context window, or a summary
pipeline. This benchmark tests the thing users actually experience: **does the agent
remember what was said, and can it answer correctly, completely, and without inventing
things — after 19–32 sessions of conversation?**

To make that fair and reproducible we filtered the upstream dataset to questions whose
answers are verifiable from the conversation text and whose evidence can be resolved, and
we score with a strict semantic judge that does not reward vague or over-precise answers.

## What it measures

| Category | Count | What it probes |
|---|---|---|
| `1` — multi-hop | 100 | Combining facts across sessions and speakers |
| `2` — temporal | 70 | Dates, order, durations, plan vs. outcome over time |
| `4` — single-hop | 10 | Direct recall of a stated fact |
| `5` — adversarial | 20 | Questions whose premise is not in the conversation — the correct answer is an abstention, not an invention |

20 questions per conversation across 10 conversations (19–32 sessions each; 5,782 turns total).

## Origin and independence

LoCoMo-200 is built from **LoCoMo** (Maharana et al., ACL 2024) —
<https://github.com/snap-research/locomo> — and is an **independent, filtered derivative**
of it. It is not affiliated with or endorsed by the LoCoMo authors; the original dataset,
its code, and any updates remain at the upstream repository. Only the subset in
`dataset/` is derived from LoCoMo (CC BY-NC 4.0 — attribution and modification notices in
`NOTICE.md`). The runner, judge, and documentation in this package are original work.

## Evaluation protocol at a glance

| | Count |
|---|---|
| Conversations | 10 |
| History sessions replayed | 272 |
| Conversation turns | 5,782 |
| Questions | 200 — 100 multi-hop, 70 temporal, 10 single-hop, 20 adversarial |
| Evaluation steps | 272 session replays + 200 question runs = **472** |

Conditions:

- **Each history replay event carries exactly one session** (speaker + text), replayed
  once, in chronological order, within that conversation.
- Questions are asked only **after all sessions of the conversation** have been replayed,
  **one question per fresh session** (a "question run"): the agent never sees the original
  transcripts again at question time — only its own memory may carry the facts.

## Quickstart

Start your agent behind the runner contract (an example baseline adapter is included):

```bash
# Example baseline: a chat model that keeps the whole transcript in context
export OPENAI_API_KEY=...
python runner/adapters/openai_chat_server.py \
  --port 8077 --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

Run the benchmark against it:

```bash
python runner/run_benchmark.py \
  --adapter-url http://localhost:8077 \
  --output results/example.results.json
```

Judge the answers:

```bash
export OPENAI_API_KEY=...
python judge/judge.py \
  --input results/example.results.json \
  --dataset dataset/locomo200.json \
  --model gpt-6-luna --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY \
  --output results/example.judge.json \
  --report-md results/example.judge.md
```

See `runner/README.md` for the adapter contract and `docs/` for methodology, scoring, and
interpretation.

## Results format and scoring

The runner writes one record per question with the model's `prediction`; the judge writes
per-claim diagnostics and aggregate scores:

- **Strict binary score** — 1 only when every required fact is present and supported;
  0 for any missing, wrong, contradictory, or materially incomplete fact.
- **Claim coverage** — fraction of required claims entailed by the answer.
- **Token F1** — lexical overlap with the reference answer.
- Breakdowns by category and by evidence difficulty (sessions involved, turn count, span,
  evidence age, query depth).

Recommended practice: judge at least **three times** and average — LLM judging varies on
borderline answers.

## Layout

```
locomo-200-benchmark/
├── dataset/        filtered dataset + selection manifest + dataset card
├── docs/           methodology, capabilities, scoring, agent contract
├── runner/         reference runner + example adapter
└── judge/          strict semantic judge (LLM-based, resumable)
```

## Licensing and attribution

- **Dataset** (`dataset/`): derived from **LoCoMo** by Maharana et al. (ACL 2024),
  licensed **CC BY-NC 4.0**. This package redistributes a *filtered subset* (200 questions;
  open-domain and unverifiable items removed — an adaptation). See `NOTICE.md` for the
  required attribution and `dataset/README.md` for the dataset card.
- **Code** (`runner/`, `judge/`): MIT (see `LICENSE`).

If you use LoCoMo or this benchmark, cite the original paper:

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
