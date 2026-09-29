# Judge

Strict semantic judging for LoCoMo-200 results. The judge reads the runner's results file,
reconstructs each question's evidence from the dataset, and asks an LLM judge to score the
answer against the reference answer **and the conversation history**.

## Run

```bash
export OPENAI_API_KEY=...
python judge/judge.py \
  --input results/example.results.json \
  --dataset dataset/locomo200.json \
  --model gpt-6-luna \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY \
  --output results/example.judge.json \
  --report-md results/example.judge.md
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--max-concurrency 4` | Parallel judge calls |
| `--timeout 180 --retries 2` | Per-call timeout and retries |
| `--resume` | Continue an existing output file |
| `--pause-after N` | Stop after N new judgments, saving a resumable checkpoint |

## Output

`*.judge.json` contains one judgment per question with the strict binary verdict
(`llm_score`), required claims and their statuses, claim coverage, token F1, and the
judge's rationale. The summary aggregates the strict score out of 100, mean claim
coverage, and breakdowns by category, evidence-session count, evidence-turn count,
evidence span, evidence age, and query depth. `*.judge.md` is a human-readable report.

## Scoring rules (summary)

A question scores **1** only when every required fact is present and supported by the
conversation; any missing, wrong, contradicted, or materially unsupported fact scores
**0**. Supported elaboration is not penalized. See `../docs/scoring.md` for details and
interpretation guidance.

## Reproducibility

- Judging is non-deterministic on borderline answers: run the judge **at least three
  times** and average for published numbers.
- The judge is resumable; use `--resume` to continue an interrupted run.
- Record the judge model and version with any published score.
