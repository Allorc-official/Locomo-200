# Runner

Drives an agent adapter through the LoCoMo-200 benchmark and writes a resumable results
file that `../judge/judge.py` scores. Standard library only — no dependencies.

## Run

```bash
python runner/run_benchmark.py \
  --adapter-url http://localhost:8077 \
  --output results/example.results.json
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--sample-id conv-26` | Run one conversation (repeatable) |
| `--question-id conv-26:qa:019` | Run one question (repeatable) |
| `--limit 20` | Cap the number of questions |
| `--skip-ingest` | Ask questions without re-sending the sessions |
| `--reset` | Send a reset before each conversation |
| `--resume` | Continue an existing output file |
| `--timeout 300 --retries 2` | Per-request timeout / retries |
| `--quiet` | Only print the final summary |

The runner saves after every question, so an interrupted run can be resumed with
`--resume`.

## Output

```json
{
  "benchmark": "locomo-200-v1",
  "adapter": {"url": "..."},
  "dataset": {"path": "...", "sha256": "..."},
  "config": {"selected_session_numbers": {"conv-26": [1, 2, "..."]}, "...": "..."},
  "samples": [
    {
      "sample_id": "conv-26",
      "questions": [
        {
          "index": "19", "question_id": "conv-26:qa:019",
          "question": "...", "expected": "dinosaurs, nature",
          "prediction": "...", "category": "1", "score": 0.0667,
          "latency_ms": 1234, "error": null
        }
      ]
    }
  ]
}
```

The `expected` field is copied verbatim from the dataset so the judge can resolve the
question's evidence; it is never sent to the adapter.

## Adapter contract

See `../docs/agent-contract.md`. A minimal example server is in
`adapters/openai_chat_server.py`:

```bash
export OPENAI_API_KEY=...
python runner/adapters/openai_chat_server.py \
  --port 8077 --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```
