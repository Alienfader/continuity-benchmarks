# Continuity Benchmarks

Reproducible benchmarks for evaluating structured-knowledge-graph retrieval in long-horizon AI agents — the architecture behind [Continuity](https://hackerware.com), with methodology comparable to MIT Media Lab's [ID-RAG paper](https://arxiv.org/abs/2509.25299) (Platnick et al., 2025).

## TL;DR findings (v6.4 walkback edition)

Cross-corpus 24-cell matrix on `data-pipeline` and `mobile-app` fixtures × {GPT-4o, Claude Sonnet 4.6} × 3 runs × 4 conditions, with paired Wilcoxon signed-rank tests on cell-level means (n=12 per contrast):

- **Retrieval >> no retrieval.** Action alignment: continuity vs baseline mean Δ +5.06 (1–10 scale), p < 0.005, **Cohen's d = 8.94**. Recall: per-question retrieval vs baseline mean Δ +0.17 cosine, **d = 11.38**.
- **Per-question targeted retrieval >> blanket (concatenated-seed) retrieval.** Recall mean Δ +0.105, p = 0.002, **d = 5.83** — the lift comes from *what* you retrieve, not *when*.
- **Injection timing does NOT matter (M2 ablation).** Holding retrieval data constant and varying only injection timing (frozen-at-session-1 vs fresh-per-session re-fire), mean Δ = −0.002 cosine, p ≈ 0.05 slightly favoring the frozen condition. 9 of 12 cells favor frozen retrieval (3 favor fresh re-fire). The "in-loop pattern is the contribution" claim is therefore not supported — the demonstrated contribution is **execution-intent-conditioned retrieval** (retrieval keyed on file paths, entities, and mutation targets extracted from agent tool calls), not the temporal pattern.
- **Inter-judge cross-validation (two studies, n=1,620 paired scores).** *Caveat: these inter-judge JSONs were generated against the v1 action-alignment runner (which had the byte-identical continuity / continuity-in-loop bug fixed in v2). The agreement statistics — judge-vs-judge correlation — are still meaningful as a robustness check on the LLM-judge methodology, but the per-condition lift ratios reflect v1 outputs. v2 re-judge is the next pre-registered step (~2h, ~$2; see Outstanding work in the v2 analysis).*
  - Paydash-api / gpt-4o-mini (n=540, Sonnet vs Gemini-2.5-flash): ρ = 0.788, κ = 0.518.
  - Cross-corpus 24-cell (n=1,080, Sonnet vs Gemini-2.5-flash): ρ = 0.716, κ = 0.559.
  - Both studies land in the same agreement band (ρ ∈ [0.72, 0.79], κ ∈ [0.52, 0.56]) — agreement reproducible across corpora, agent models, and run windows. Continuity > baseline preserved in every cell under both judges (Sonnet 2.30× lift in ratio of overall means; Gemini 1.80×).

Full numbers, methodology, M2 ablation: **[`reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md`](reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md)** and the v6.4 white paper.

---

## Quickstart

```bash
git clone https://github.com/alienfader/continuity-benchmarks
cd continuity-benchmarks
npm install
cp .env.example .env
# edit .env — add at minimum ANTHROPIC_API_KEY (judge) and OPENAI_API_KEY (agent)

# 1. Smoke tests (no API spend, ~10–15 seconds each — verify install + paths):
npm run test:smoke              # 3-condition recall-over-time on paydash, mock model
npm run test:smoke-v2           # 4-condition recall (v2 M2 ablation conditions) on data-pipeline
npm run test:smoke-alignment    # action-alignment, mock model + mock judge

# 2. Single real-model invocation (~5–10 min, ~$0.10 with gpt-4o-mini):
npm run bench:recall -- --fixture paydash-api --model gpt-4o-mini --seed 1

# 3. Full v2 cross-corpus matrix (~12–14h wall, ~$25–30 at Tier 2 OpenAI):
npm run bench:matrix-v2
# resume failed cells (idempotent, skips ones whose JSON already exists):
npm run bench:matrix-v2-resume

# 4. Re-judge action-alignment outputs with Gemini (cross-validation):
npm run rejudge:paydash         # n=540 paydash inter-judge
npm run rejudge:cross-corpus    # n=1,080 v2 cross-corpus inter-judge

# 5. Replicate the v2 analysis (no API spend, ~5 seconds — paired Wilcoxon + Cohen's d):
npm run analyze:v2
```

---

## What's in this repo

| Path | Purpose |
|---|---|
| `fixtures/paydash-api/` | Sanitized fictional Express+Postgres project with 19 architectural decisions in `.continuity/decisions.json`. The benchmark target. |
| `prompts/quizzes/paydash-api.json` | 20 recall-quiz questions paired with ground-truth answers, used by `recall-over-time`. |
| `prompts/quizzes/{ml-platform,mobile-app,data-pipeline,infra-platform}.json` | Four extra fixtures for cross-domain extension (no raw outputs in this release). |
| `runners/recall-over-time.ts` | 7-session multi-turn drift benchmark, mirrors the ID-RAG identity-recall protocol. Wires the 4-condition v2 matrix (`baseline`, `continuity-blanket`, `continuity-perq-frontloaded`, `continuity-in-loop`). |
| `runners/action-alignment.ts` | 30 prompts × 3 conditions; LLM-as-a-judge scores proposed actions 1–10. |
| `runners/convergence-time.ts` | Time-to-completion benchmark for fixed multi-step refactor tasks. |
| `runners/head-to-head.ts` | Continuity-vs-MemPalace 50-query benchmark (§4.3 of the white paper). Imports `@continuity/core`'s `SemanticSearchService`, which is closed-source — see file header for the dependency note + how to swap in your own retrieval engine. Saved per-query results are at `reports/head-to-head-*.json`. |
| `runners/middleware-replay.ts` | `continuity-mcp-middleware` runner — end-to-end replay through the production Continuity MCP server. Three retrieval modes: `mcp-search` (single-shot, calls `search_decisions` directly), `agent-loop` (2-turn, agent decides whether/how to query `search_decisions`), `auto-middleware` (2-turn, agent uses the benchmark-mode `bash` tool; AutoRetrievalMiddleware fires server-side and injects decisions into `_meta.relevantDecisions`). Set `CONTINUITY_MCP_PATH` to your local MCP server bundle. Smoke: `npm run test:smoke-middleware-replay` / `test:smoke-agent-loop` / `test:smoke-auto-middleware`. |
| `runners/shared/mcp-client.ts` | MCP client wiring — spawns the production server as a stdio subprocess, dispatches arbitrary tool calls (`dispatchToolCall`), parses tool-result `_meta.relevantDecisions` payload, plus convenience helpers for `search_decisions` and middleware-firing tool calls. |
| `runners/shared/agent-client.ts` | Tool-calling agent abstraction (Anthropic + OpenAI + deterministic Mock). Powers the 2-turn loops in `agent-loop` and `auto-middleware` modes. Supports one round of tool use per question (turn 1: agent decides what tool to call; turn 2: agent reads tool result and generates final answer). |
| `runners/re-judge.py` | Cross-validates saved paydash action-alignment scores using Gemini-2.5-flash; emits Cohen's κ + Spearman ρ. |
| `runners/re-judge-cross-corpus.py` | Same but for the v2 cross-corpus matrix (n=1,080 paired scores). |
| `runners/bootstrap-ci.py` | BCa (bias-corrected and accelerated) bootstrap 95% CIs on Cohen's d for every contrast in the v2 matrix; 10,000 resamples. Outputs `reports/id-rag-parity-v2/bootstrap-ci.json`. |
| `runners/experimental-gaps-analysis-v2.py` | Paired Wilcoxon signed-rank tests and Cohen's d for the v2 matrix. |
| `runners/shared/` | LLM provider clients, BM25 retrieval + entity-extraction key, cosine eval embeddings, noise generator. |
| `reports/id-rag-parity/` | All 11 successful per-run JSON outputs (paydash × {gpt-4o, gpt-4o-mini} × 3 runs × 2 runners) + the 540-pair Sonnet-vs-Gemini inter-judge JSON. |
| `reports/id-rag-parity-v2/` | v2 cross-corpus 24-cell matrix (data-pipeline + mobile-app × {gpt-4o, claude-sonnet-4-6} × 3 runs × 4 conditions), the M2-ablation analysis, the 1,080-pair cross-corpus inter-judge JSON, and `bootstrap-ci.json` (BCa CIs on Cohen's d). |
| `reports/id-rag-parity-summary.md` | Synthesized findings, methodology, comparison with ID-RAG. |

---

## Methodology

### Recall-over-time (mirrors ID-RAG)

Per project × model × condition (`baseline` / `continuity` / `continuity-in-loop`):
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

**What the in-loop runner actually does vs. the production middleware:**

| Step | Production middleware (`AutoRetrievalMiddleware`) | Public `continuity-in-loop` runner |
|---|---|---|
| Trigger | Agent issues `Bash`/`Edit`/`Write` tool call | Runner receives the prompt for a quiz question |
| Key extraction | Pulls file paths + entities from tool-call **arguments** | Pulls file paths + entities from the **prompt text** (`extractEntities`) |
| Retrieval | Queries decision store on extracted keys | BM25 over decision Q+A+tags on extracted keys |
| Delivery | Injects matched decisions into tool result `_meta.relevantDecisions` | Prepends matched decisions to the agent prompt |

The runner faithfully simulates the **retrieval-keying logic** of the middleware (entity extraction → query → top-K decisions). It does **not** exercise the **tool-call interception delivery mechanism** (prompt-prepend vs `_meta`-inject is a real distinction the public benchmark does not currently isolate). The §4.7 timing-ablation conclusions in the white paper therefore apply to entity-keyed retrieval delivered via prompt-prepend.

Production-middleware delivery is implemented end-to-end in `runners/middleware-replay.ts` (smoke-tests pass against the production MCP server in all three modes). To run it: set `CONTINUITY_MCP_PATH=/path/to/packages/mcp-server/dist/index.js` and invoke `npm run bench:middleware-replay -- --fixture <name> --model <name> --retrieval=<mode> --output reports/<dir>`. Three retrieval modes:

- `--retrieval=mcp-search` (single-shot): calls the production `search_decisions` MCP tool with the prompt as the query. Tests the production retrieval ranker (`SemanticSearchService` RRF hybrid: semantic + keyword + tags) delivered via real MCP. No agent reasoning. Works on any fixture.

- `--retrieval=agent-loop` (2-turn): the agent has `search_decisions` advertised as a tool. Turn 1: agent decides whether/how to query. Turn 2: MCP returns decisions, agent generates the final answer. Tests the **full production agent + retrieval ranker via real MCP**, which is what most production agent operators actually exercise. Works on any fixture.

- `--retrieval=auto-middleware` (2-turn): the agent has the benchmark-mode `bash` tool advertised. Turn 1: agent issues `cat <path>`. Server-side: `AutoRetrievalMiddleware` extracts paths from the bash command, looks them up in `code-links.json`, and injects matched decisions into the tool result's `_meta.relevantDecisions` field. Turn 2: agent reads the tool result + injected decisions, generates the final answer. Tests the **production AutoRetrievalMiddleware delivery shape end-to-end**, which is the specific code path the §4.7 in-loop simulation only approximated.

  Requirements for `auto-middleware`:
  - The MCP server must be built with the `bash` tool registered. The runner sets `CONTINUITY_BENCHMARK_MODE=1` on the spawned server automatically; this gates the `bash` tool registration to benchmark use only (the env var is unset in normal user-facing deployments).
  - The workspace must have `.continuity/code-links.json` mapping decision IDs to file paths the agent can reference. The `data-pipeline` fixture ships with one pre-authored.

Populating any of these as a row in the v2 matrix costs ~$30–50 in API spend per mode; doing so is the next pre-registered protocol expansion.

---

## Reproducing the headline numbers

```bash
# Full matrix: 12 invocations, ~4 hours wall clock, ~$15–25 in OpenAI spend
for model in gpt-4o gpt-4o-mini; do
  for run in 1 2 3; do
    for runner in recall-over-time action-alignment; do
      mkdir -p reports/my-run/paydash-api/$model/run-$run
      npx ts-node runners/$runner.ts \
        --fixture paydash-api --model $model --seed $run \
        --output reports/my-run/paydash-api/$model/run-$run/$runner
    done
  done
done

# Cross-validate the judge:
python3 runners/re-judge.py

# Compare your numbers against ours:
diff <(jq -S . reports/my-run/.../recall-over-time.json) \
     <(jq -S . reports/id-rag-parity/.../recall-over-time.json)
```

### Known issues

- **OpenAI Tier 1 throttles gpt-4o at 3 RPM.** First burst of any invocation will hit a 429. Subsequent invocations succeed once calls self-pace below the limit. Add prepayment credits + verify Tier 2 if you want full-speed runs.
- **Local Ollama Qwen2.5-7B is too slow on M1** (1+ hour per condition). Use a hosted Qwen API (Together AI, Groq) if you need that lane.
- **Sonnet judge sometimes returns `overloaded_error`.** Single-invocation failure; retry the runner and it should succeed on the next attempt.

---

## Comparison with ID-RAG

| Dimension | ID-RAG (MIT, Sept 2025) | Continuity (this repo) |
|---|---|---|
| Domain | Persona identity in multi-agent simulations | Architectural decisions in a software project |
| Knowledge graph | Chronicle (17 nodes, 16 edges per agent) | paydash decisions.json (19 decisions) |
| Models tested | GPT-4o, GPT-4o-mini, Qwen2.5-7B | GPT-4o, GPT-4o-mini |
| Runs per condition | 4 | 3 |
| Sessions / timesteps | 7 | 7 |
| Baseline recall (cosine, t=7) | 0.51–0.56 | 0.51–0.52 |
| With-RAG recall (cosine, t=7) | 0.58–0.60 | 0.59 (continuity) / 0.69 (in-loop) |

The lift we observe is **of comparable magnitude** to ID-RAG's reported lift, on a different domain. Absolute recall numbers are not strictly apples-to-apples across the two papers (different ground truths, different retrieval targets, different embedding models for scoring); within-paper condition comparisons are the defensible axis.

---

## What's NOT in this repo

- The Continuity VS Code extension itself — see [hackerware.continuity-ultimate on the VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=hackerware.continuity-ultimate).
- The production `AutoRetrievalMiddleware` source itself. This repo's `continuity-in-loop` runner faithfully simulates the middleware's **retrieval-keying logic** (entity extraction → query → top-K decisions) but delivers the matched decisions by prompt-prepend rather than by injecting into tool-result `_meta.relevantDecisions`. The end-to-end production-middleware replay is implemented in `runners/middleware-replay.ts` (calls the production MCP server's `search_decisions` and/or `bash` tools through the real stdio transport) — set `CONTINUITY_MCP_PATH` to a local clone of the commercial workspace to use it.
- Any proprietary decision data from real projects. Only the sanitized fictional fixtures are here.
- A working `@continuity/core` install. `runners/head-to-head.ts` is vendored as a reference implementation but `npm install` does not include the closed-source `@continuity/core` package; see the file's header note for how to swap in your own retrieval engine.

---

## Citation

```bibtex
@misc{continuity-benchmarks-2026,
  title  = {Continuity Benchmarks: Structured Knowledge-Graph Retrieval for Long-Horizon AI Agents},
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
