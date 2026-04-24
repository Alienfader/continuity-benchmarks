# Continuity Benchmarks

Reproducible benchmarks for evaluating structured-knowledge-graph retrieval in long-horizon AI agents — the architecture behind [Continuity](https://hackerware.com), with methodology comparable to MIT Media Lab's [ID-RAG paper](https://arxiv.org/abs/2509.25299) (Rahnama et al., 2025).

## TL;DR findings

Run on `paydash-api` fixture (19 architectural decisions) × {GPT-4o, GPT-4o-mini} × 3 runs per condition:

- **Baseline → Continuity lifts action-alignment 3×** (GPT-4o: 2.82 → 8.77 on 1–10 scale)
- **In-loop retrieval lifts recall-over-time even further:** session-7 mean cosine 0.519 (baseline) / 0.600 (continuity) / 0.693 (in-loop) for GPT-4o. Fraction of questions clearing 0.7 cosine **doubles** under in-loop (23% → 55%).
- **Inter-judge cross-validation (Sonnet vs Gemini-2.5-flash, n=540):** Spearman ρ = 0.788 strong, Cohen's κ = 0.518 moderate. Headline findings direction-robust across judges.

Full numbers, methodology, and per-section interpretation: **[`reports/id-rag-parity-summary.md`](reports/id-rag-parity-summary.md)**.

---

## Quickstart

```bash
git clone https://github.com/alienfader/continuity-benchmarks
cd continuity-benchmarks
npm install
cp .env.example .env
# edit .env — add at minimum ANTHROPIC_API_KEY (judge) and OPENAI_API_KEY (agent)

# 1. Smoke test with a mock LLM (no API spend, ~10 seconds):
npm run test:smoke

# 2. Single real-model invocation (~5–10 min, ~$0.10):
npx ts-node runners/recall-over-time.ts --fixture paydash-api --model gpt-4o-mini --seed 1

# 3. Re-validate Sonnet's judge scores against Gemini (~25 min, ~$3):
python3 runners/re-judge.py
```

---

## What's in this repo

| Path | Purpose |
|---|---|
| `fixtures/paydash-api/` | Sanitized fictional Express+Postgres project with 19 architectural decisions in `.continuity/decisions.json`. The benchmark target. |
| `prompts/quizzes/paydash-api.json` | 20 recall-quiz questions paired with ground-truth answers, used by `recall-over-time`. |
| `prompts/quizzes/{ml-platform,mobile-app,data-pipeline,infra-platform}.json` | Four extra fixtures for cross-domain extension (no raw outputs in this release). |
| `runners/recall-over-time.ts` | 7-session multi-turn drift benchmark, mirrors the ID-RAG identity-recall protocol. |
| `runners/action-alignment.ts` | 30 prompts × 3 conditions; LLM-as-a-judge scores proposed actions 1–10. |
| `runners/convergence-time.ts` | Time-to-completion benchmark for fixed multi-step refactor tasks. |
| `runners/re-judge.py` | Cross-validates saved action-alignment scores using Gemini-2.5-flash; emits Cohen's κ + Spearman ρ. |
| `runners/shared/` | LLM provider clients, BM25 + RRF retrieval, cosine eval embeddings, noise generator. |
| `reports/id-rag-parity/` | All 11 successful per-run JSON outputs (paydash × {gpt-4o, gpt-4o-mini} × 3 runs × 2 runners). |
| `reports/id-rag-parity/inter-judge.json` | 540 blinded Sonnet–Gemini score pairs with κ + ρ statistics. |
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
- **`continuity`** — top-K relevant decisions retrieved once and prepended to the prompt
- **`continuity-in-loop`** — same retrieval, plus a second augmentation pass using the top decision's tags (simulates `AutoRetrievalMiddleware`'s "re-fire with context" behavior)

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
- The `AutoRetrievalMiddleware` implementation — this repo measures the architectural pattern, not the specific middleware. The runners' `continuity-in-loop` condition is a faithful behavioral simulation (re-fire retrieval with augmented context).
- Any proprietary decision data from real projects. Only the sanitized fictional `paydash-api` fixture is here.
- The `head-to-head.ts` MemPalace comparison — depends on `@continuity/core`'s `SemanticSearchService`. May be vendored in a follow-up release.

---

## Citation

```bibtex
@misc{continuity-benchmarks-2026,
  title  = {Continuity Benchmarks: Structured Knowledge-Graph Retrieval for Long-Horizon AI Agents},
  author = {Hackerware},
  year   = {2026},
  url    = {https://github.com/alienfader/continuity-benchmarks}
}
```

And the methodology paper this work extends:

```bibtex
@misc{rahnama2025idrag,
  title  = {ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents},
  author = {Rahnama, Hossein and others},
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
