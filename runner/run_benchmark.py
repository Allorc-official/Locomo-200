#!/usr/bin/env python3
"""LoCoMo-200 benchmark runner — drive any agent through a JSON HTTP contract.

Sends each conversation's sessions in order, then asks the selected questions, and
writes a resumable results file that the judge in ``../judge/`` can score.

Adapter contract (see ../docs/agent-contract.md):

    POST {"type": "health"}                          -> {"status": "ok"}      (optional)
    POST {"type": "reset", "sample_id": ...}         -> {"status": "ok"}      (with --reset)
    POST {"type": "ingest_session", ...}             -> {"status": "ok"}
    POST {"type": "answer_question", ...}            -> {"answer": "..."}

Usage:

    python runner/run_benchmark.py \
        --adapter-url http://localhost:8077 \
        --output results/example.results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_ID = "locomo-200-v1"
SESSION_KEY = re.compile(r"^session_(\d+)$")
_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post(url: str, payload: dict[str, Any], *, timeout: float, retries: int) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
            try:
                return json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                return {"answer": body.strip()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {detail}"
        except Exception as exc:  # noqa: BLE001 - adapter errors must not abort the run
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(last_error or "adapter request failed")


def _answer_from(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        for key in ("answer", "text", "response", "output", "content"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (dict, list)):
                nested = _answer_from(value)
                if nested:
                    return nested
    return ""


def _normalise(text: Any) -> str:
    value = "" if text is None else str(text).lower()
    value = _WORD.sub(" ", value)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def _token_f1(expected: Any, prediction: str) -> float:
    gold = _normalise(expected).split()
    guess = _normalise(prediction).split()
    if not gold or not guess:
        return 1.0 if gold == guess else 0.0
    gold_counts, guess_counts = Counter(gold), Counter(guess)
    overlap = sum((gold_counts & guess_counts).values())
    if not overlap:
        return 0.0
    precision = overlap / len(guess)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _sessions(conversation: dict[str, Any]) -> list[tuple[int, str | None, list[dict[str, Any]]]]:
    numbers = sorted(int(m.group(1)) for key in conversation if (m := SESSION_KEY.match(str(key))))
    return [
        (number, conversation.get(f"session_{number}_date_time"), conversation.get(f"session_{number}") or [])
        for number in numbers
    ]


def _turns(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = " ".join(str(turn.get("text") or "").split())
        if not text:
            continue
        payload.append({"speaker": str(turn.get("speaker") or "unknown"), "text": text})
    return payload


def _save(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LoCoMo-200 benchmark against an agent adapter.")
    default_dataset = Path(__file__).resolve().parents[1] / "dataset" / "locomo200.json"
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Path to locomo200.json")
    parser.add_argument("--adapter-url", required=True, help="Agent adapter endpoint (JSON POST)")
    parser.add_argument("--output", type=Path, required=True, help="Results file to write (resumable)")
    parser.add_argument("--sample-id", action="append", default=[], help="Limit to a conversation (repeatable)")
    parser.add_argument("--question-id", action="append", default=[], help="Limit to a question id (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to run")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per request on failure")
    parser.add_argument("--reset", action="store_true", help="Send a reset before each conversation")
    parser.add_argument("--skip-ingest", action="store_true", help="Ask questions without sending sessions")
    parser.add_argument("--resume", action="store_true", help="Continue an existing output file")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary")
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()

    samples = dataset
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [s for s in samples if str(s.get("sample_id")) in wanted]
        missing = wanted - {str(s.get("sample_id")) for s in samples}
        if missing:
            parser.error(f"unknown --sample-id values: {', '.join(sorted(missing))}")
    if args.question_id:
        wanted_questions = set(args.question_id)
        filtered = []
        for sample in samples:
            kept = [q for q in sample.get("qa", []) if str(q.get("question_id")) in wanted_questions]
            if kept:
                filtered.append({**sample, "qa": kept})
        samples = filtered
        missing = wanted_questions - {
            str(q.get("question_id")) for sample in samples for q in sample.get("qa", [])
        }
        if missing:
            parser.error(f"unknown --question-id values: {', '.join(sorted(missing))}")

    remaining = args.limit
    for sample in samples:
        if remaining is None:
            break
        kept = sample.get("qa", [])[:remaining]
        sample["qa"] = kept
        remaining -= len(kept)

    total_questions = sum(len(s.get("qa", [])) for s in samples)
    if total_questions == 0:
        parser.error("no questions selected")

    output_path = args.output.expanduser().resolve()
    document: dict[str, Any] | None = None
    recorded: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.is_file():
        document = json.loads(output_path.read_text(encoding="utf-8"))
        for sample in document.get("samples", []):
            for question in sample.get("questions", []):
                recorded[str(question.get("question_id"))] = question

    if document is None:
        document = {
            "benchmark": BENCHMARK_ID,
            "generated_at": _now(),
            "adapter": {"url": args.adapter_url},
            "dataset": {"path": str(dataset_path), "sha256": dataset_sha},
            "config": {"selected_session_numbers": {}, "selected_question_ids": []},
            "samples": [],
        }

    try:
        health = _post(args.adapter_url, {"type": "health"}, timeout=10.0, retries=0)
        if not args.quiet:
            print(f"[health] {json.dumps(health)[:200]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[health] warning: {exc}", flush=True)

    by_sample = {str(sample.get("sample_id")): sample for sample in document.get("samples", [])}
    done = 0
    failures = 0
    started = time.perf_counter()

    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        sessions = _sessions(sample.get("conversation") or {})
        document["config"]["selected_session_numbers"][sample_id] = [n for n, _, _ in sessions]
        result_sample = by_sample.get(sample_id)
        if result_sample is None:
            result_sample = {"sample_id": sample_id, "questions": []}
            document["samples"].append(result_sample)
            by_sample[sample_id] = result_sample

        if not args.skip_ingest:
            if args.reset:
                _post(
                    args.adapter_url,
                    {"type": "reset", "benchmark": BENCHMARK_ID, "sample_id": sample_id},
                    timeout=args.timeout,
                    retries=args.retries,
                )
                if not args.quiet:
                    print(f"[{sample_id}] reset", flush=True)
            for number, date_time, turns in sessions:
                payload = {
                    "type": "ingest_session",
                    "benchmark": BENCHMARK_ID,
                    "sample_id": sample_id,
                    "session_number": number,
                    "date_time": date_time,
                    "speakers": [
                        str((sample.get("conversation") or {}).get("speaker_a") or ""),
                        str((sample.get("conversation") or {}).get("speaker_b") or ""),
                    ],
                    "turns": _turns(turns),
                }
                _post(args.adapter_url, payload, timeout=args.timeout, retries=args.retries)
                if not args.quiet:
                    print(f"[{sample_id}] session {number}/{len(sessions)} ingested", flush=True)

        for question in sample.get("qa", []):
            question_id = str(question.get("question_id"))
            existing = recorded.get(question_id)
            if existing is not None and existing.get("prediction") not in (None, ""):
                continue
            started_q = time.perf_counter()
            error: str | None = None
            answer = ""
            try:
                response = _post(
                    args.adapter_url,
                    {
                        "type": "answer_question",
                        "benchmark": BENCHMARK_ID,
                        "sample_id": sample_id,
                        "question_id": question_id,
                        "category": question.get("category"),
                        "question": question.get("question"),
                    },
                    timeout=args.timeout,
                    retries=args.retries,
                )
                answer = _answer_from(response)
                if not answer:
                    error = "adapter returned an empty answer"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            latency_ms = round((time.perf_counter() - started_q) * 1000, 1)
            record = {
                "index": question.get("source_qa_index"),
                "question_id": question_id,
                "question": question.get("question"),
                "expected": question.get("answer"),
                "prediction": answer,
                "category": question.get("category"),
                "category_name": question.get("category_name"),
                "answerability": question.get("answerability"),
                "score": round(_token_f1(question.get("answer"), answer), 4),
                "latency_ms": latency_ms,
                "error": error,
            }
            result_sample["questions"] = [
                q for q in result_sample["questions"] if str(q.get("question_id")) != question_id
            ]
            result_sample["questions"].append(record)
            recorded[question_id] = record
            done += 1
            if error:
                failures += 1
            document["config"]["selected_question_ids"] = sorted(recorded)
            _save(output_path, document)
            if not args.quiet or error:
                status = "error" if error else "ok"
                print(
                    f"[{sample_id}] question {done}/{total_questions} {question_id} "
                    f"status={status} latency_ms={latency_ms}",
                    flush=True,
                )

    document["generated_at"] = _now()
    document["summary"] = {
        "questions": done,
        "errors": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    _save(output_path, document)
    print(
        f"[done] questions={done} errors={failures} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
