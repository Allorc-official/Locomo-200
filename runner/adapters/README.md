# Example adapters

`openai_chat_server.py` is a minimal reference implementation of the adapter contract: it
keeps every ingested session in memory and answers questions by sending the full
transcript to an OpenAI-compatible chat completions endpoint.

It exists to make the benchmark runnable in one command and to document the contract by
example. It is **not** a good memory system — the full transcript must fit the model's
context window, and long conversations will be truncated (see `--max-transcript-chars`).

```bash
export OPENAI_API_KEY=...
python runner/adapters/openai_chat_server.py \
  --port 8077 --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

Replace it with an adapter for your own agent. A complete adapter is tiny: implement the
four request types (`health`, `reset`, `ingest_session`, `answer_question`) and return
`{"answer": "..."}`. Any language or stack works — the contract is JSON over HTTP.
