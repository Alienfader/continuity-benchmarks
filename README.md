# continuity-benchmarks

Reproducible benchmarks for **structured-knowledge retrieval in long-horizon AI coding agents**.

The repo asks a single question:

> Does retrieval keyed on what the agent is *about to do* improve the correctness of its actions, more than retrieval keyed on the user's prompt?

Three runners measure this against fictional codebases with hand-authored architectural decisions, plus the public LongMemEval dataset for an external cross-check. Everything is bring-your-own-API-keys and reproducible from `npm` / `python` scripts.

---

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/Alienfader/continuity-benchmarks.git
cd continuity-benchmarks
npm install

# 2. Set API keys (pick whichever benchmarks you want to run)
cp .env.example .env
# edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...   # required for Sonnet judge / agent
#   OPENAI_API_KEY=sk-proj-...     # required for GPT-4o agent / judge
#   GOOGLE_API_KEY=AIzaSy-...      # required for Gemini Flash / inter-judge

# 3. Run a smoke test (no API calls, uses mock client)
npm run test:smoke

# 4. Run the smallest real benchmark (~$0.10, ~13 min, Gemini Flash)
#    See LongMemEval section below for setup.
```

---

## Benchmarks

| Runner | Question it answers | Output | Cost |
|---|---|---|---|
| `runners/action-alignment.ts` | Does the agent obey architectural constraints when proposing an action? | LLM-judge 1–10 across 30 prompts × 3 conditions per cell | ~$1 / cell |
| `runners/recall-over-time.ts` | Does it retain decisions across 7 noisy sessions? | Cosine similarity vs ground truth × 4 conditions per cell | ~$1 / cell |
| `runners/longmemeval.ts` | Does it work on the public LongMemEval-S benchmark? | Per-question accuracy (Gemini autoeval, optional GPT-4o re-judge) | ~$0.10 / 50 questions |

### ID-RAG matrix (fixtures + recall + alignment)

```bash
# Single-cell smoke test (mock model, no API calls)
npm run test:smoke-v2

# Full v2 cross-corpus matrix: 2 fixtures × 2 models × 3 runs × 2 runners = 24 cells
# Wall: ~14h. Cost: ~$25-30. Set ANTHROPIC_API_KEY + OPENAI_API_KEY.
npm run bench:matrix-v2

# After the matrix completes, run the analysis script
npm run analyze:v2

# Optional: inter-judge replication with Gemini-2.5-flash (~2h, ~$2)
npm run rejudge:cross-corpus-v2
```

The matrix runners have **overwrite guards** — re-running against a populated `reports/id-rag-parity-v2/` is a no-op by default. Set `FORCE=1` to override.

### LongMemEval-S subsample (Gemini Flash, free-tier-friendly)

```bash
# 1. Download the raw dataset (280 MB) from HuggingFace
curl -sSLo fixtures/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 2. Generate a deterministic 50-question stratified subsample
python3 scripts/longmemeval-subsample.py

# 3. Run the benchmark (~13 min wall, free tier on Gemini Flash)
npm run bench:longmemeval -- \
  --sample fixtures/longmemeval/sample-50.json \
  --output reports/longmemeval/run-1

# 4. Optional: re-judge with GPT-4o for leaderboard-comparable numbers (~4 min, ~$1)
npm run rejudge:longmemeval -- --run-dir reports/longmemeval/run-1
```

See `fixtures/longmemeval/README.md` for the data setup details + caveats versus the official LongMemEval leaderboard.

### Custom retrieval system (bring your own)

The `systems/` directory accepts adapters that plug into the same fixtures + scoring scripts. Drop an adapter at `systems/<name>/index.ts` exporting a `RetrievalSystem` (one `init(decisions) → Retriever` function), then:

```bash
npm run bench:custom -- --runner=recall --system=my-adapter --fixture=data-pipeline --model=gpt-4o-mini
npm run bench:compare    # produces side-by-side comparison report
```

See `systems/README.md` for the adapter contract and an example BM25 implementation.

---

## What the benchmarks have found

From the v2 cross-corpus matrix (n=12 cells, paired Wilcoxon):

- **Retrieval >> no retrieval.** Action alignment Cohen's d = 8.94. Recall d = 11.38.
- **Per-question targeted retrieval >> blanket retrieval.** Recall d = 5.83 — the lift comes from *what* you retrieve, not *when*.
- **Injection timing does not matter** (M2 ablation). Holding retrieval data constant and varying only timing: d = -0.68, CI crosses zero.
- **Cross-judge robustness.** Sonnet ↔ Gemini-2.5-flash on n=1,080 paired scores: ρ = 0.722, κ = 0.558 (substantial agreement). Direction preserved across judges.

From the LongMemEval-S subsample (n=50, GPT-4o re-judged):

- **Baseline (no context): 14% accuracy. Continuity (BM25 top-5): 66% accuracy.** +52 pp lift.
- Inter-judge (Gemini Flash vs GPT-4o) agreement 90%, κ = 0.774.

Full numbers + methodology: `reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md`.

---

## Repository layout

```
fixtures/                Hand-authored fictional projects + LongMemEval data
  paydash-api/             v1 baseline (Express + Postgres + Redis)
  data-pipeline/           v2 (Kafka + Snowflake + Dagster)
  mobile-app/              v2 (React Native + iOS/Android)
  ml-platform/             v2 (training infra) — fixtures only, no benchmarks yet
  infra-platform/          v2 (cloud infra) — fixtures only, no benchmarks yet
  longmemeval/             External: xiaowu0162/longmemeval-cleaned subsamples

prompts/quizzes/         20-question recall quizzes per fixture
runners/                 TypeScript + Python benchmark runners
  shared/                  BM25 retriever, LLM clients, fixture loader
scripts/                 Bash drivers + Python subsampling
systems/                 Bring-your-own retrieval adapters
reports/                 Output directory (gitignored — regenerate locally)
docs/methodology.md      Protocol details for each runner
docs/reproducibility.md  End-to-end re-run instructions
```

---

## Citing this work

```bibtex
@misc{continuity-benchmarks-2026,
  author = {Continuity contributors},
  title  = {Continuity Benchmarks: Execution-Intent Memory Evaluation for AI Coding Agents},
  year   = {2026},
  url    = {https://github.com/Alienfader/continuity-benchmarks}
}
```

## License

MIT. Fixtures, runners, and scripts are all freely usable for academic and commercial work.

LongMemEval fixture data is downloaded at runtime from [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (MIT licensed, separate from this repo).
