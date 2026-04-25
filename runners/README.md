# ID-RAG-Parity Runners

Three benchmark runners that adapt the ID-RAG Parallel methodology of [Platnick et al. (2025)](https://arxiv.org/abs/2509.25299) to Continuity's coding-agent decision store. They produce the numbers reported in §4 of the v6.4 write-up.

## What's here

| Runner | Measures | Paper analog |
|---|---|---|
| [`recall-over-time.ts`](./recall-over-time.ts) | Cosine-sim recall across 7 sessions, 3 conditions | Identity recall |
| [`action-alignment.ts`](./action-alignment.ts) | LLM judge 1–10 on 30 proposed actions | Behavioral alignment |
| [`convergence-time.ts`](./convergence-time.ts) | Steps / tokens / wall-clock for a 5-step refactor | Task convergence (not run for the v6.4 numbers) |
| [`re-judge.py`](./re-judge.py) | Re-scores saved action-alignment runs with a second LLM judge | Inter-judge validation (§4.4) |

Shared utilities in [`./shared/`](./shared/):
- `llm-providers.ts` — lightweight OpenAI / Anthropic / Ollama / Mock clients
- `eval-embeddings.ts` — `all-mpnet-base-v2` via `@xenova/transformers`
- `noise-generator.ts` — deterministic ~5k-token off-topic filler
- `retrieval.ts` — BM25 retriever for the Continuity conditions
- `fixtures.ts` — loader for the `paydash-api` fixture and quizzes
- `cli.ts` — argument parser shared across the three runners

## Quickstart

From the repo root:

```bash
# Recall-over-time on paydash-api, mock LLM (no keys needed, end-to-end safe)
npx ts-node runners/recall-over-time.ts \
  --fixture paydash-api --model mock

# With real GPT-4o-mini (requires OPENAI_API_KEY in env)
npx ts-node runners/recall-over-time.ts \
  --fixture paydash-api --model gpt-4o-mini

# Action alignment, smaller prompt count for smoke tests
npx ts-node runners/action-alignment.ts \
  --fixture paydash-api --model mock --actions 10

# Convergence time
npx ts-node runners/convergence-time.ts \
  --fixture paydash-api --model mock

# Re-judge a saved action-alignment run with Gemini 2.5 Flash (requires GOOGLE_API_KEY)
python3 runners/re-judge.py
```

## Supported models

Pass one of these to `--model`:
- `mock` — deterministic stub, no network (default; safe for CI)
- `gpt-4o` / `gpt-4o-mini` — OpenAI via REST (needs `OPENAI_API_KEY`)
- `qwen2.5-7b` — local Ollama (needs `ollama serve` + the model pulled). Too slow on M1 to fit the v6.4 time budget; use a hosted Qwen API for production runs.
- `claude-sonnet-4-5` / `claude-sonnet-4-6` — Anthropic SDK (needs `ANTHROPIC_API_KEY`)

## Conditions

Every runner takes `--conditions <csv>` (default: all three). CLI flags are the names used in code; the v6.4 paper uses the descriptive names in parentheses.

- `baseline` (no memory) — agent has no project decisions in its prompt
- `continuity` (Passive) — top-K decisions retrieved once at session start, prepended to the prompt; no retrieval tool exposed during the session
- `continuity-in-loop` (In-Loop) — middleware injects matched decisions on every file-touching step, simulating `AutoRetrievalMiddleware`'s tool-call hook

The Passive vs In-Loop distinction is **not** about retrieval quality (the two are statistically tied on single-prompt benchmarks) — it's about *when* retrieval fires. In-Loop wins specifically on multi-session recall where the agent would otherwise fail to query for relevant decisions. See `reports/id-rag-parity-summary.md` §3 for the full breakdown.

## Reports

Each runner writes `<base>.json` and `<base>.md` to the path you pass via `--output <base>`. The v6.4 outputs live under `reports/id-rag-parity/paydash-api/<model>/run-<n>/`.

## Tests

Shared utilities have `node:test`-based unit tests:

```bash
npx ts-node runners/shared/__tests__/run-all.ts
```

The tests do not depend on jest, do not require network access, and do not require `@xenova/transformers` to be preloaded (embedder tests use a deterministic hash-based `MockEmbedder`).

## Design notes

- **LLM abstraction is standalone.** The production `LLMProviderManager` is VS Code-bound. The benchmark clients talk to each provider directly so the runners work in plain Node.
- **Retrieval is BM25-only inside the benchmark.** Keeps the benchmark transparent and independent of the MiniLM embedding cache. The production middleware uses BM25 + a file-name index fused via reciprocal rank fusion (RRF, k=60).
- **Noise is deterministic.** Same `--seed` → same noise blocks → reproducible runs.

## Full matrix

The full v6.4 matrix is `paydash-api × {gpt-4o, gpt-4o-mini} × 3 runs × 2 runners` (~4 hours wall-clock, ~$15–25 in OpenAI spend at Tier 2). See `README.md` at the repo root for the loop. Two extra fixtures (ml-platform, infra-platform) and a third model lane (Qwen2.5-7B) were dropped from this run for time; quiz files for the extra fixtures ship under `prompts/quizzes/` and can be plugged into the same runners.
