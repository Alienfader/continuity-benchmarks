# ID-RAG-Parallel Runners

Three benchmark runners that mirror the MIT ID-RAG paper's methodology, adapted
for Continuity's architectural-decision memory rather than identity graphs.

## What's here

| Runner | Measures | Paper analog |
|---|---|---|
| [`recall-over-time.ts`](./recall-over-time.ts) | Cosine-sim drift across 7 sessions, 3 conditions | Identity recall |
| [`action-alignment.ts`](./action-alignment.ts) | Claude-Sonnet judge 1-10 on 30 proposed actions | Behavioral alignment |
| [`convergence-time.ts`](./convergence-time.ts) | Steps / tokens / wall-clock for a 5-step refactor | Task convergence |

Shared utilities in [`./shared/`](./shared/):
- `llm-providers.ts` — lightweight OpenAI / Anthropic / Ollama / Mock clients
- `eval-embeddings.ts` — `all-mpnet-base-v2` via `@xenova/transformers`
- `noise-generator.ts` — deterministic ~5k-token off-topic filler
- `retrieval.ts` — BM25 retriever for the Continuity conditions
- `fixtures.ts` — loader for Clio's fixtures + quizzes (legacy-paydash fallback)
- `cli.ts` — argument parser shared across the three runners

## Quickstart

From the repo root:

```bash
# Recall-over-time on paydash-api, mock LLM (no keys needed, end-to-end safe)
npx ts-node benchmarks/src/id-rag-parallel/runners/recall-over-time.ts \
  --fixture paydash-api --model mock

# With real GPT-4o-mini (requires OPENAI_API_KEY in env)
npx ts-node benchmarks/src/id-rag-parallel/runners/recall-over-time.ts \
  --fixture paydash-api --model gpt-4o-mini

# Action alignment, smaller prompt count for smoke tests
npx ts-node benchmarks/src/id-rag-parallel/runners/action-alignment.ts \
  --fixture paydash-api --model mock --actions 10

# Convergence time
npx ts-node benchmarks/src/id-rag-parallel/runners/convergence-time.ts \
  --fixture paydash-api --model mock
```

## Supported models

Pass one of these to `--model`:
- `mock` — deterministic stub, no network (default; safe for CI)
- `gpt-4o` / `gpt-4o-mini` — OpenAI via REST (needs `OPENAI_API_KEY`)
- `qwen2.5-7b` — local Ollama (needs `ollama serve` + the model pulled)
- `claude-sonnet-4-5` / `claude-sonnet-4-6` — Anthropic SDK (needs `ANTHROPIC_API_KEY`)

## Conditions

Every runner takes `--conditions <csv>` (default: all three):
- `baseline` — no project decisions in the prompt
- `continuity` — top-K decisions retrieved once upfront, prepended to every prompt
- `continuity-in-loop` — decisions re-retrieved before each step or question

## Reports

Each runner writes `<base>.json` and `<base>.md` to `benchmarks/reports/`.
Override with `--output <base>` to rename.

## Tests

Shared utilities have `node:test`-based unit tests:

```bash
npx ts-node benchmarks/src/id-rag-parallel/runners/shared/__tests__/run-all.ts
```

The tests do not depend on jest, do not require network access, and do not
require `@xenova/transformers` to be preloaded (embedder tests use a
deterministic hash-based `MockEmbedder`).

## Design notes

- **LLM abstraction is standalone.** The production `LLMProviderManager` is VS
  Code-bound. The benchmark clients talk to each provider directly so the
  runners work in plain Node.
- **Retrieval is BM25-only.** Keeps the benchmark transparent and independent
  of the MiniLM embedding cache. Easy to swap for `SemanticSearchService` later.
- **Noise is deterministic.** Same `--seed` → same noise blocks → reproducible
  runs.
- **Fixtures have a legacy fallback.** If Clio's `benchmarks/src/id-rag-parallel/fixtures/`
  hasn't been merged yet, `paydash-api` falls back to
  `demo-projects/peer-review/with-continuity/.continuity/decisions.json`.

## Full matrix

Running the full ID-RAG parallel matrix (5 fixtures × 4 models × 3 conditions × 3 runners)
is left to the user — it costs real API money. The runners are scoped to prove the
methodology on one fixture + one model end-to-end.
