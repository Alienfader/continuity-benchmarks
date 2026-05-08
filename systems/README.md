# Bring-your-own retrieval system

This directory is the contribution lane for **alternative memory systems** — long-context, RAG pipelines, vector DBs, agent-framework memory layers, custom retrieval engines. Drop an adapter here, run the same fixtures + scoring scripts, get a comparable result row.

## Adapter contract

A system is a directory `systems/<name>/` containing `index.ts` (or `index.js`) that **default-exports** a `RetrievalSystem`:

```ts
import type { RetrievalSystem } from '../../runners/shared/system-adapter';
import type { Decision } from '../../runners/shared/fixtures';
import type { Retriever } from '../../runners/shared/retrieval';

const myAdapter: RetrievalSystem = {
  name: 'my-vector-db',
  description: 'Pinecone with text-embedding-3-large + cosine top-K',

  async init(decisions: Decision[]): Promise<Retriever> {
    // 1. Bootstrap your store (embed, index, spin up sidecar, etc.)
    const index = await embedAndIndex(decisions);

    // 2. Return a Retriever — { retrieve(query, k): Decision[] }
    return {
      retrieve(query: string, k: number): Decision[] {
        return index.queryTopK(query, k);
      },
    };
  },
};

export default myAdapter;
```

That's the whole contract: a function that takes a query string + a top-K and returns up to K `Decision` objects from the fixture's corpus.

## Running your adapter

```bash
# Against the recall-over-time runner (multi-session drift):
npm run bench:custom -- \
  --system=my-vector-db \
  --runner=recall \
  --fixture=paydash-api \
  --model=gpt-4o-mini \
  --seed=1 \
  --output=reports/my-run/recall

# Against the action-alignment runner (1–10 LLM-judge):
npm run bench:custom -- \
  --system=my-vector-db \
  --runner=alignment \
  --fixture=paydash-api \
  --model=gpt-4o-mini \
  --seed=1 \
  --output=reports/my-run/alignment

# Compare your numbers against the baseline:
npm run bench:compare -- \
  --baseline=reports/id-rag-parity-v2 \
  --custom=reports/my-run \
  --output=reports/summary.json
```

`bench:custom` runs the underlying runner with the `continuity-in-loop` condition replaced by your adapter's retriever. The other conditions (`baseline`, `continuity-blanket`, `continuity-perq-frontloaded`) still run via the built-in BM25, so the resulting JSON is directly comparable.

## What gets passed to your adapter

- `decisions: Decision[]` — array of `{ id, question, answer, tags, ... }`. Live shape lives in `runners/shared/fixtures.ts`.
- The runner calls `init(decisions)` once per (fixture, model, run) combination. After that it calls `retrieve(query, k)` repeatedly.

## What your adapter is judged on

- **Recall** — cosine similarity vs ground-truth answers across 7 multi-session quizzes (`runners/recall-over-time.ts`)
- **Action alignment** — LLM-judge score 1–10 on 30 proposed-action prompts (`runners/action-alignment.ts`)

Same scoring, same fixtures, same seeds as the built-in conditions. No hidden evaluation logic — see [`reports/id-rag-parity-summary.md`](../reports/id-rag-parity-summary.md) and [`reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md`](../reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md).

## Reference adapters

| `systems/<name>/` | What it shows |
|---|---|
| [`example-bm25/`](./example-bm25/) | Minimal adapter wrapping the built-in `BM25Retriever` — the smallest possible adapter that runs end-to-end. Good starting template. |

## Contributing your results

1. Fork the repo
2. Add `systems/<your-system>/`
3. Run the smoke tests: `npm run bench:custom -- --system=<your-system> --runner=recall --model=mock --output=/tmp/smoke-custom`
4. Run the real-model headline lanes (recall + alignment, fixture=paydash-api, model=gpt-4o-mini, 3 seeds)
5. Open a PR with the run JSONs under `reports/<your-system>/` and a one-paragraph methodology note

The maintainers will run an independent replication with their own API keys before merging — all PRs land with reproducibility numbers attached.
