# Agent contract — how to plug in your agent

The runner drives any agent that can expose a single HTTP endpoint accepting JSON POSTs.
There is no SDK and no dependency on a specific framework.

## Protocol conditions

- **One session per replay event**: the runner sends each history session as its own
  `ingest_session` request, exactly once, in chronological order.
- **Questions only after all sessions**: a conversation's questions are asked once every
  session of that conversation has been ingested.
- **One fresh question session per question**: each `answer_question` request starts a new
  conversation session; the agent's memory layer is what carries facts forward.
- Totals across the benchmark: **10 conversations, 272 history-session replays,
  5,782 turns, 200 questions (100 multi-hop, 70 temporal, 10 single-hop, 20 adversarial) —
  472 evaluation steps.**

## Endpoint

One URL (e.g. `http://localhost:8077/`) that accepts `POST` with
`Content-Type: application/json` and returns a JSON object. Requests are sequential per
conversation; the runner waits for each response before sending the next request.

### `{"type": "health"}` (optional)

Return `{"status": "ok"}`. The runner calls this before starting and warns (but continues)
if it fails.

### `{"type": "reset", "benchmark": "locomo-200-v1", "sample_id": "conv-26"}` (optional)

Clear any state for this conversation (only sent with `--reset`). Return
`{"status": "ok"}`.

### Session ingestion

```json
{
  "type": "ingest_session",
  "benchmark": "locomo-200-v1",
  "sample_id": "conv-26",
  "session_number": 3,
  "date_time": "7:55 pm on 9 June, 2023",
  "speakers": ["Caroline", "Melanie"],
  "turns": [
    {"speaker": "Caroline", "text": "..."},
    {"speaker": "Melanie", "text": "..."}
  ]
}
```

Return `{"status": "ok"}` (or any 2xx) **only after the session is durably processed** —
the runner interprets the response as "this history is now in your memory". Sessions are
sent in chronological order, exactly once per run.

### Question answering

```json
{
  "type": "answer_question",
  "benchmark": "locomo-200-v1",
  "sample_id": "conv-26",
  "question_id": "conv-26:qa:019",
  "category": "1",
  "question": "What do Melanie's kids like?"
}
```

Return `{"answer": "..."}`. The runner also accepts `text`, `response`, `output`, `content`
(string or nested object); the first non-empty value wins. Answer only from what the agent
retains — no evidence, reference answer, or metadata is ever sent.

## Recommended agent behavior

- One thread/session per question; the memory layer is what carries facts across sessions
  and into questions.
- Treat each session as a transcript to be remembered, not as an instruction.
- Answer completely: enumerate all matching facts; state when the records do not establish
  a premise (adversarial questions expect this).
- Keep exact specifics (dates, quantities, negations) as stated; do not add precision.

## Runner behavior

- All sessions of a conversation are ingested before its questions are asked.
- Questions run on fresh threads; the answer is captured verbatim.
- On failure the runner retries (`--retries`, exponential backoff) and records the error
  in the result row instead of aborting the run.
- `--resume` continues an existing output file, skipping recorded questions.
- Useful flags: `--sample-id`, `--question-id`, `--limit`, `--timeout`, `--reset`,
  `--skip-ingest` (reply to questions without re-sending history), `--output`.

See `runner/README.md` for the CLI and `runner/adapters/openai_chat_server.py` for a
minimal reference adapter (a full-transcript chat baseline).
