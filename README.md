# Continuity: Execution-Intent Memory Benchmark for AI Coding Agents

Most AI memory systems are evaluated on **semantic recall**.

Continuity evaluates something stricter:

> *Can an AI coding agent consistently follow architectural decisions while actively modifying a real codebase?*

---

## 🧪 Why this exists

AI coding agents typically fail in a specific way:

- They **retrieve** the correct information
- But still **violate it** during file-level edits

This happens because memory is usually:

- **Optional** — the agent must choose to query it
- **Semantic** — embedding similarity, decoupled from execution
- **Decoupled from tool usage** — retrieval runs against the prompt, not against what the agent is *about to do*

Continuity tests whether **execution-intent-conditioned retrieval** — retrieval keyed on file paths, entities, and mutation targets extracted from the agent's tool calls — improves behavioral correctness under mutation.

---

## 🔬 What this benchmark measures

Three runners, each isolating one axis of memory failure:

| Runner | What it measures | Output |
|---|---|---|
| `runners/action-alignment.ts` | Does the agent obey architectural constraints when proposing actions? | LLM-judge score 1–10 across 30 prompts × 3 conditions |
| `runners/recall-over-time.ts` | Does it retain decisions across noisy multi-session windows? | Cosine similarity vs ground truth across 7 sessions × 4 conditions |
| `runners/middleware-replay.ts` | Does end-to-end production middleware deliver decisions through real MCP tool calls? | 3 retrieval modes: `mcp-search`, `agent-loop`, `auto-middleware` |

---

## 📊 TL;DR findings (v6.4 walkback edition)

Cross-corpus 24-cell matrix on `data-pipeline` and `mobile-app` fixtures × {GPT-4o, Claude Sonnet 4.6} × 3 runs × 4 conditions, paired Wilcoxon signed-rank tests on cell-level means (n=12 per contrast):

- **Retrieval >> no retrieval.** Action alignment: continuity vs baseline mean Δ +5.06 (1–10 scale), p < 0.005, **Cohen's d = 8.94**. Recall: per-question retrieval vs baseline mean Δ +0.17 cosine, **d = 11.38**.
- **Per-question targeted retrieval >> blanket (concatenated-seed) retrieval.** Recall mean Δ +0.105, p = 0.002, **d = 5.83** — the lift comes from *what* you retrieve, not *when*.
- **Injection timing does NOT matter (M2 ablation).** Holding retrieval data constant and varying only injection timing (frozen-at-session-1 vs fresh-per-session re-fire), mean Δ = −0.002 cosine, p ≈ 0.05. 9 of 12 cells favor frozen retrieval (3 favor fresh re-fire). The "in-loop pattern is the contribution" claim is therefore **not supported** — the demonstrated contribution is **execution-intent-conditioned retrieval** (retrieval keyed on file paths, entities, and mutation targets extracted from agent tool calls), **not the temporal pattern**.
- **Inter-judge cross-validation (two studies, n=1,620 paired scores).** *Caveat: these inter-judge JSONs were generated against the v1 action-alignment runner (which had the byte-identical continuity / continuity-in-loop bug fixed in v2). Judge-vs-judge correlation is still a meaningful methodology robustness check, but per-condition lift ratios reflect v1 outputs. v2 re-judge is the next pre-registered step (~2h, ~$2; see Outstanding work in the v2 analysis).*
  - Paydash-api / gpt-4o-mini (n=540, Sonnet vs Gemini-2.5-flash): ρ = 0.788, κ = 0.518.
  - Cross-corpus 24-cell (n=1,080, Sonnet vs Gemini-2.5-flash): ρ = 0.716, κ = 0.559.
  - Both studies land in the same agreement band (ρ ∈ [0.72, 0.79], κ ∈ [0.52, 0.56]) — agreement reproducible across corpora, agent models, and run windows. Continuity > baseline preserved in every cell under both judges (Sonnet 2.30× lift in ratio of overall means; Gemini 1.80×).

Full numbers, methodology, M2 ablation: **[`reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md`](reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md)** and the v6.4 white paper.

---

## 🧠 Core hypothesis

> Retrieval correctness matters less than retrieval **triggering condition**.

Specifically:

- semantic similarity ≠ correct behavior
- long context ≠ consistency
- memory ≠ constraint adherence

Continuity tests whether **execution-conditioned retrieval improves behavioral correctness under mutation**.

---

## 🚀 Quickstart

```bash
git clone https://github.com/Alienfader/continuity-benchmarks
cd continuity-benchmarks
npm install
cp .env.example .env
# edit .env — add at minimum ANTHROPIC_API_KEY (judge) and OPENAI_API_KEY (agent)

# 1. Smoke tests (no API spend, ~10–15 seconds each):
npm run test:smoke              # 3-condition recall on paydash, mock model
npm run test:smoke-v2           # 4-condition recall (v2 M2 ablation conditions) on data-pipeline
npm run test:smoke-alignment    # action-alignment, mock model + mock judge

# 2. Single real-model invocation (~5–10 min, ~$0.10 with gpt-4o-mini):
npm run bench:recall -- --fixture paydash-api --model gpt-4o-mini --seed 1
npm run bench:alignment -- --fixture paydash-api --model gpt-4o-mini --seed 1

# 3. Full v2 cross-corpus matrix (~12–14h wall, ~$25–30 at Tier 2 OpenAI):
npm run bench:matrix-v2
npm run bench:matrix-v2-resume   # idempotent, skips cells already on disk

# 4. Re-judge action-alignment outputs with Gemini (cross-validation):
npm run rejudge:paydash         # n=540 paydash inter-judge
npm run rejudge:cross-corpus    # n=1,080 v2 cross-corpus inter-judge

# 5. Replicate the v2 analysis (no API spend, ~5 seconds — paired Wilcoxon + Cohen's d):
npm run analyze:v2
npm run bootstrap:ci             # BCa 95% CIs on Cohen's d, 10,000 resamples
```

---

## ⚔️ Compare your system

If you ship "memory for coding agents" — long-context, RAG, vector DB, agent framework, custom — you can run it through the same fixtures.

The benchmark is **three layers**, easiest to hardest:

### 1. `systems/` adapter — drop-in retrieval replacement (Recommended)

Implement a `RetrievalSystem` and the runner does the rest:

```ts
// systems/my-vector-db/index.ts
import type { RetrievalSystem } from '../../runners/shared/system-adapter';

export default {
  name: 'my-vector-db',
  description: 'Pinecone + text-embedding-3-large + cosine top-K',
  async init(decisions) {
    const index = await embedAndIndex(decisions);
    return { retrieve: (query, k) => index.queryTopK(query, k) };
  },
} satisfies RetrievalSystem;
```

```bash
npm run bench:custom -- --runner=recall    --system=my-vector-db --fixture=paydash-api --model=gpt-4o-mini --output=reports/my-run/recall
npm run bench:custom -- --runner=alignment --system=my-vector-db --fixture=paydash-api --model=gpt-4o-mini --output=reports/my-run/alignment
npm run bench:compare -- --baseline=reports/id-rag-parity-v2/.../recall-over-time.json --custom=reports/my-run/recall.json --output=reports/my-summary.json
```

Full contract + reference adapter (`systems/example-bm25/`) at [`systems/README.md`](systems/README.md). Smoke test the contract with `npm run test:smoke-custom` (no API spend, ~10s).

### 2. End-to-end MCP middleware lane

`runners/middleware-replay.ts` speaks the Model Context Protocol — point it at any MCP server that exposes a retrieval tool by setting `CONTINUITY_MCP_PATH=/path/to/your/server.js`. Three replay modes: `mcp-search` (single-shot tool call), `agent-loop` (2-turn agent decides whether/how to query), `auto-middleware` (server-side middleware extracts retrieval keys from tool-call arguments).

### 3. Direct runner invocation

The recall + alignment runners (`runners/recall-over-time.ts`, `runners/action-alignment.ts`) work with any model. Fixture decisions live at `fixtures/<name>/.continuity/decisions.json`; quizzes at `prompts/quizzes/<name>.json`. Use this if you need to change the conditions themselves rather than just the retriever.

Note: `runners/head-to-head.ts` (Continuity vs MemPalace, §4.3 of the white paper) imports the closed-source `@continuity/core` `SemanticSearchService` as the production retrieval ranker — it's a reference implementation, not a contribution lane.

---

## 📦 What's in this repo

| Path | Purpose |
|---|---|
| `fixtures/paydash-api/` | Sanitized fictional Express+Postgres project with 19 architectural decisions in `.continuity/decisions.json`. Headline benchmark target. |
| `fixtures/{ml-platform,mobile-app,data-pipeline,infra-platform}/` | Four extra fixtures for cross-domain extension. `data-pipeline` and `mobile-app` are the v2 cross-corpus matrix corpora. |
| `prompts/quizzes/<fixture>.json` | 20 recall-quiz questions paired with ground-truth answers. |
| `runners/recall-over-time.ts` | 7-session multi-turn drift benchmark. Wires the 4-condition v2 matrix (`baseline`, `continuity-blanket`, `continuity-perq-frontloaded`, `continuity-in-loop`). |
| `runners/action-alignment.ts` | 30 prompts × 3 conditions; LLM-as-a-judge scores proposed actions 1–10. |
| `runners/convergence-time.ts` | Time-to-completion benchmark for fixed multi-step refactor tasks. |
| `runners/head-to-head.ts` | Continuity-vs-MemPalace 50-query benchmark (§4.3). Imports closed-source `@continuity/core::SemanticSearchService`. |
| `runners/middleware-replay.ts` | End-to-end replay through a real MCP server. Three retrieval modes (`mcp-search`, `agent-loop`, `auto-middleware`). |
| `runners/shared/mcp-client.ts` | MCP client wiring — spawns the production server as a stdio subprocess, parses `_meta.relevantDecisions`. |
| `runners/shared/agent-client.ts` | Tool-calling agent abstraction (Anthropic + OpenAI + deterministic Mock). |
| `runners/re-judge.py` / `re-judge-cross-corpus.py` | Cross-validate saved action-alignment scores using Gemini-2.5-flash. |
| `runners/bootstrap-ci.py` | BCa bootstrap 95% CIs on Cohen's d for every contrast in the v2 matrix; 10,000 resamples. |
| `runners/experimental-gaps-analysis-v2.py` | Paired Wilcoxon + Cohen's d for the v2 matrix. |
| `reports/id-rag-parity/` | v1 paydash + cross-corpus per-run JSONs + n=540 paydash inter-judge JSON + n=1,080 cross-corpus inter-judge JSON. |
| `reports/id-rag-parity-v2/` | v2 cross-corpus 24-cell matrix + M2 ablation analysis + bootstrap CIs. |
| `reports/id-rag-parity-summary.md` | Synthesized findings, methodology, comparison with ID-RAG. |

---

## 🔬 Methodology

### Recall-over-time (mirrors ID-RAG)

Per project × model × condition (`baseline` / `continuity-blanket` / `continuity-perq-frontloaded` / `continuity-in-loop`):
- Run 7 sessions sequentially
- Inject ~5k tokens of unrelated noise between sessions
- At each session boundary, ask all 20 quiz questions
- Score each answer with cosine similarity vs ground truth using `all-mpnet-base-v2`

### Action-alignment

For 30 proposed-action prompts × 3 conditions:
- Capture the agent's proposed action under each condition
- LLM judge (Claude Sonnet 4.6) scores 1–10 for "does this action align with the project's decisions?"
- Optional: re-judge with Gemini-2.5-flash to cross-validate

### Conditions explained

- **`baseline`** — agent has no project context in its prompt
- **`continuity`** — top-K decisions retrieved once on the full prompt as the seed query, prepended to the agent prompt (single-shot, full-prompt-keyed retrieval — the "passive RAG" condition)
- **`continuity-blanket`** *(recall-over-time only)* — top-K retrieved once using the concatenation of all 20 quiz-question stems as the seed; the same blob is prepended to every session
- **`continuity-perq-frontloaded`** *(recall-over-time only, M2-ablation comparand)* — per-question retrieval computed ONCE at session 1, the same 20 question-specific blobs re-injected unchanged at every session boundary
- **`continuity-in-loop`** — retrieval keyed on **entities extracted from the prompt** (file paths, capitalized identifiers, tech terms — see `runners/shared/retrieval.ts::extractEntities`). For action-alignment this is single-shot entity-keyed retrieval; for recall-over-time the per-question retrieval is re-fired FRESH at every session boundary

### What the in-loop runner actually does vs. the production middleware

| Step | Production middleware (`AutoRetrievalMiddleware`) | Public `continuity-in-loop` runner |
|---|---|---|
| Trigger | Agent issues `Bash`/`Edit`/`Write` tool call | Runner receives the prompt for a quiz question |
| Key extraction | Pulls file paths + entities from tool-call **arguments** | Pulls file paths + entities from the **prompt text** (`extractEntities`) |
| Retrieval | Queries decision store on extracted keys | BM25 over decision Q+A+tags on extracted keys |
| Delivery | Injects matched decisions into tool result `_meta.relevantDecisions` | Prepends matched decisions to the agent prompt |

The runner faithfully simulates the **retrieval-keying logic** of the middleware. Production-middleware delivery is implemented end-to-end in `runners/middleware-replay.ts` (smoke-tests pass against the production MCP server in all three modes).

---

## 📦 Reproducibility

- Fixed evaluation fixtures (5 sanitized fictional projects)
- Deterministic scoring scripts (paired Wilcoxon, Cohen's d, BCa bootstrap)
- Open prompts, open quizzes, open conditions
- Raw per-run outputs included under `reports/`
- No hidden evaluation logic

---

## 📈 Reproducing the headline numbers

```bash
# Full v2 cross-corpus matrix: 24 invocations, ~12–14h wall clock, ~$25–30 OpenAI
npm run bench:matrix-v2

# Cross-validate the judge:
npm run rejudge:cross-corpus

# Run the analysis:
npm run analyze:v2
npm run bootstrap:ci

# Compare your numbers against ours:
diff <(jq -S . reports/my-run/.../recall-over-time.json) \
     <(jq -S . reports/id-rag-parity-v2/.../recall-over-time.json)
```

### Known issues

- **OpenAI Tier 1 throttles gpt-4o at 3 RPM.** First burst of any invocation will hit a 429. Subsequent invocations succeed once calls self-pace below the limit.
- **Local Ollama Qwen2.5-7B is too slow on M1** (1+ hour per condition). Use a hosted Qwen API (Together AI, Groq) if you need that lane.
- **Sonnet judge sometimes returns `overloaded_error`.** Single-invocation failure; retry the runner.

---

## 📊 Comparison with ID-RAG

| Dimension | ID-RAG (MIT, Sept 2025) | Continuity (this repo) |
|---|---|---|
| Domain | Persona identity in multi-agent simulations | Architectural decisions in a software project |
| Knowledge graph | Chronicle (17 nodes, 16 edges per agent) | paydash decisions.json (19 decisions) |
| Models tested | GPT-4o, GPT-4o-mini, Qwen2.5-7B | GPT-4o, GPT-4o-mini, Claude Sonnet 4.6 |
| Runs per condition | 4 | 3 |
| Sessions / timesteps | 7 | 7 |
| Baseline recall (cosine, t=7) | 0.51–0.56 | 0.51–0.52 (paydash v1) |
| With-RAG recall (cosine, t=7) | 0.58–0.60 | 0.60 (continuity) / 0.69 (in-loop), paydash v1 |

The lift we observe is **of comparable magnitude** to ID-RAG's reported lift, on a different domain. Absolute recall numbers are not strictly apples-to-apples (different ground truths, different retrieval targets, different embedding models for scoring); within-paper condition comparisons are the defensible axis.

---

## 🔗 What's NOT in this repo

- The Continuity VS Code extension itself — see [hackerware.continuity-ultimate on the VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=hackerware.continuity-ultimate).
- The production `AutoRetrievalMiddleware` source. The end-to-end production-middleware replay is implemented in `runners/middleware-replay.ts` (calls a real MCP server through real stdio transport) — set `CONTINUITY_MCP_PATH` to a local clone of the commercial workspace to use it.
- Any proprietary decision data from real projects. Only sanitized fictional fixtures.
- A working `@continuity/core` install. `runners/head-to-head.ts` is vendored as a reference implementation — `npm install` does not include the closed-source `@continuity/core` package.

---

## ⚠️ Positioning

This is **not** a leaderboard for LLM performance.

It is a **behavioral correctness benchmark for AI coding agents under mutation pressure**.

If your system claims "memory for coding agents" — long-context, RAG, vector DB, agent framework, custom — you can run it through the same fixtures and the same scoring scripts. The fixtures, the prompts, the conditions, the scoring scripts, and the raw outputs are all in this repo.

---

## 📣 Benchmark transparency

- No hidden prompts
- No private evaluation set
- All scoring scripts included
- All outputs reproducible locally

---

## 📌 Citation

```bibtex
@misc{continuity-benchmarks-2026,
  title  = {Continuity Benchmarks: Execution-Intent Memory for Long-Horizon AI Coding Agents},
  author = {Goncalves, Thiago},
  year   = {2026},
  url    = {https://github.com/alienfader/continuity-benchmarks}
}
```

And the methodology paper this work extends:

```bibtex
@misc{platnick2025idrag,
  title  = {ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents},
  author = {Platnick, Daniel and others},
  year   = {2025},
  eprint = {2509.25299},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2509.25299}
}
```

---

## Contact

Academic inquiries, replication support, benchmark extensions: **contact@hackerware.com**
Press / partnership: **media@hackerware.com**

## License

MIT. See `LICENSE`.
