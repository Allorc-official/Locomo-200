#!/usr/bin/env python3
"""Reference agent adapter — full-transcript chat baseline (example only).

Implements the LoCoMo-200 adapter contract (see ../../docs/agent-contract.md) with the
simplest possible agent: it keeps every ingested session in memory and, at question time,
sends the transcript plus the question to an OpenAI-compatible chat completions endpoint.

This is a *baseline*, not a good memory system: the whole transcript must fit the model's
context window. Replace it with your own agent to evaluate real memory behavior.

    python runner/adapters/openai_chat_server.py \
      --port 8077 --model gpt-4o-mini \
      --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STATE: dict[str, list[str]] = {}
CONFIG: dict[str, Any] = {}

DEFAULT_SYSTEM_PROMPT = (
    "You are answering questions about a long conversation you have been given. "
    "Answer using only facts supported by the conversation. "
    "Be complete: include every relevant fact, keep dates, quantities, and negations "
    "exactly as stated, and do not add inferred details or rankings. "
    "If the conversation does not establish the premise of the question, say so plainly "
    "instead of substituting a nearby fact. Answer directly; do not mention the records."
)


def _chat(messages: list[dict[str, str]]) -> str:
    base_url = str(CONFIG["base_url"]).rstrip("/")
    api_key = os.getenv(str(CONFIG["api_key_env"]), "")
    if not api_key:
        raise RuntimeError(f"environment variable {CONFIG['api_key_env']} is not set")
    body = json.dumps(
        {
            "model": CONFIG["model"],
            "temperature": CONFIG["temperature"],
            "messages": messages,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(CONFIG["timeout"])) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"chat endpoint HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("chat endpoint returned no choices")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def _answer(payload: dict[str, Any]) -> str:
    sample_id = str(payload.get("sample_id") or "")
    question = str(payload.get("question") or "")
    transcript = "\n".join(STATE.get(sample_id, []))
    max_chars = int(CONFIG["max_transcript_chars"])
    if len(transcript) > max_chars:
        transcript = "[earlier transcript truncated]\n" + transcript[-max_chars:]
    user = f"Conversation transcript:\n{transcript}\n\nQuestion: {question}"
    return _chat(
        [
            {"role": "system", "content": str(CONFIG["system_prompt"])},
            {"role": "user", "content": user},
        ]
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health"}:
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._respond(400, {"error": "invalid JSON body"})
            return
        try:
            kind = str(payload.get("type") or "")
            if kind == "health":
                self._respond(200, {"status": "ok"})
                return
            if kind == "reset":
                STATE.pop(str(payload.get("sample_id") or ""), None)
                self._respond(200, {"status": "ok"})
                return
            if kind == "ingest_session":
                sample_id = str(payload.get("sample_id") or "")
                lines = STATE.setdefault(sample_id, [])
                lines.append(f"Session {payload.get('session_number')} — {payload.get('date_time')}")
                for turn in payload.get("turns") or []:
                    lines.append(f"{turn.get('speaker')}: {turn.get('text')}")
                self._respond(200, {"status": "ok"})
                return
            if kind == "answer_question":
                self._respond(200, {"answer": _answer(payload)})
                return
            self._respond(400, {"error": f"unknown request type: {kind}"})
        except Exception as exc:  # noqa: BLE001
            self._respond(500, {"error": str(exc)[:500]})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Example full-transcript adapter for LoCoMo-200.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-transcript-chars", type=int, default=60_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    CONFIG.update(vars(args))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[adapter] listening on http://{args.host}:{args.port} model={args.model}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
