# Continuity Benchmarks

Reproducible benchmark artifacts for [Continuity](https://hackerware.com), an in-loop retrieval pattern for persistent project rationale in AI coding agents.

This repo accompanies the v6.4 write-up *Continuity: An In-Loop Retrieval Pattern for Persistent Project Rationale in AI Coding Agents* (Goncalves, 2026). Methodology adapts the **ID-RAG Parallel** runners introduced by [Platnick et al. (2025)](https://arxiv.org/abs/2509.25299) for the coding-agent domain. The in-loop placement of retrieval inside the agent's decision loop is from that paper; Continuity is a domain-specific engineering adaptation, not a novel mechanism.

## TL;DR findings

Run on `paydash-api` fixture (19 architectural decisions) × {GPT-4o, GPT-4o-mini} × 3 runs per condition:

- **Exposing decisions to the agent at all matters most.** Both retrieval conditions lift single-prompt action alignment ~3× over a no-memory baseline (GPT-4o: 2.82 → 8.77 on 1–10 scale; GPT-4o-mini: 2.88 → 8.22).
- **In-loop injection ≈ passive retrieval on single-prompt benchmarks.** When the relevant decisions are already served inline with each prompt, automatic re-firing is redundant. Action-alignment scores tie within run-to-run variance (8.77 Passive vs 8.61 In-Loop on GPT-4o), and the head-to-head retrieval comparison shows the same pattern (44 vs 43 wins).
- **In-loop injection clearly wins on multi-session recall.** Fraction of recall questions clearing a 0.7 cosine threshold roughly doubles on both models (23% → 55% on GPT-4o; 28% → 50% on GPT-4o-mini). The benefit is *coverage* — the agent sees decisions it would not have thought to query for — not *drift reduction* (the fixture is too short to drift).
- **Inter-judge cross-validation (Sonnet vs Gemini-2.5-flash, n=540):** Spearman ρ = 0.788 strong, Cohen's κ = 0.518 moderate. Headline directions hold under both judges; absolute scores differ in calibration (Gemini is +1.44 points more generous on average).

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
| `runners/recall-over-time.ts` | 7-session multi-turn recall benchmark; mirrors the ID-RAG Parallel identity-recall protocol. |
| `runners/action-alignment.ts` | 30 prompts × 3 conditions; LLM-as-judge scores proposed actions 1–10. |
| `runners/convergence-time.ts` | Time-to-completion benchmark for fixed multi-step refactor tasks (not run for the v6.4 numbers). |
| `runners/re-judge.py` | Cross-validates saved action-alignment scores using Gemini-2.5-flash; emits Cohen's κ + Spearman ρ. |
| `runners/shared/` | LLM provider clients, BM25 + RRF retrieval, cosine eval embeddings, noise generator. |
| `reports/id-rag-parity/` | All 11 successful per-run JSON outputs (paydash × {gpt-4o, gpt-4o-mini} × 3 runs × 2 runners). |
| `reports/id-rag-parity/inter-judge.json` | 540 blinded Sonnet–Gemini score pairs with κ + ρ statistics. |
| `reports/id-rag-parity-summary.md` | Synthesized findings, methodology, and the relationship to Platnick et al. (2025). |

---

## Methodology

### Recall-over-time (adapts ID-RAG Parallel)

Per project × model × condition (`baseline` / `continuity` / `continuity-in-loop`):
- Run 7 sessions sequentially in a single context (no reset between sessions)
- Inject ~5k tokens of unrelated noise between sessions
- At each session boundary, ask all 20 quiz questions
- Score each answer with cosine similarity vs ground truth using `all-mpnet-base-v2`

### Action-alignment

For 30 proposed-action prompts × 3 conditions:
- Capture the agent's proposed action under each condition
- LLM judge (Claude Sonnet 4.6) scores 1–10 for "does this action align with the project's decisions?"
- Optional: re-judge with Gemini-2.5-flash to cross-validate (see §4.4 in the write-up)

### Conditions explained

The CLI flags below are the names used in code; the v6.4 paper uses the more descriptive names in parentheses.

- **`baseline` (no memory)** — agent has no project context in its prompt
- **`continuity` (Passive)** — top-K relevant decisions retrieved once at session start and prepended to the prompt; no retrieval tool exposed during the session
- **`continuity-in-loop` (In-Loop)** — middleware injects matched decisions into every file-touching tool result's `_meta` block, keyed on affected paths (the pattern described in §3 of the write-up)

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
- **Local Ollama Qwen2.5-7B is too slow on M1** (1+ hour per condition). Use a hosted Qwen API (Together AI, Groq) if you need that lane. The v6.4 numbers are GPT-4o / GPT-4o-mini only.
- **Sonnet judge sometimes returns `overloaded_error`.** Single-invocation failure; retry the runner and it should succeed on the next attempt. One GPT-4o-mini recall-over-time run failed this way and is reported as n=2 instead of n=3.

---

## Relationship to ID-RAG (Platnick et al., 2025)

| Dimension | ID-RAG (MIT, Sept 2025) | Continuity (this repo) |
|---|---|---|
| Domain | Persona identity in multi-agent simulations | File-scoped architectural decisions in coding agents |
| Knowledge structure | Identity knowledge graph (17 nodes, 16 edges per agent) | Flat decision store keyed on file path (19 decisions in fixture) |
| Retrieval trigger | Inside the agent's decision loop, before action selection | Inside the agent's tool-execution loop, on file-touching tool calls |
| Models tested | GPT-4o, GPT-4o-mini, Qwen2.5-7B | GPT-4o, GPT-4o-mini |
| Runs per condition | 4 | 3 |
| Sessions / timesteps | 7 | 7 |
| Headline finding | 19% / 58% convergence reduction; arrests persona drift | 3× action-alignment lift from exposing decisions; ~2× recall coverage from in-loop trigger |

**The structural idea — placing retrieval inside the agent's decision loop rather than leaving it to the agent's discretion — is from Platnick et al.** Earlier drafts of the Continuity write-up claimed independent discovery of this pattern; that claim is withdrawn in v6.4. What this repo benchmarks is the engineering adaptation of that pattern to the coding-agent domain, where the trigger is a file-touching tool call rather than a decision-loop step, and the store is a flat path-keyed decision log rather than a persona knowledge graph.

Absolute recall numbers across the two studies are not apples-to-apples (different ground truth, different retrieval target, same embedding scale). What is comparable is the magnitude of the lift, which is of the same order in both.

---

## Limitations

The v6.4 write-up's §6 lists these caveats; reproducing them here for repo-level visibility:

- **Single project fixture.** All numbers are from `paydash-api` (19 decisions). Two other planned fixtures (ml-platform, infra-platform) were dropped for time. Generalization to larger decision counts, other languages, and non-synthetic projects is untested.
- **Two base models from one vendor.** GPT-4o and GPT-4o-mini. Qwen2.5-7B was attempted but did not complete within the time budget on local Ollama.
- **LLM-judge scores without human validation.** Inter-judge agreement (§4.4) addresses LLM-judge agreement, not human–LLM agreement. A human-labeled gold subset is the most useful next validation.
- **File-scoped decisions only.** The pattern does not handle cross-cutting constraints well, and we do not claim it does.
- **Drift-reduction claim is withdrawn.** All three conditions are flat across 7 sessions on this fixture (drift slopes < 0.003/session). The in-loop benefit in §4.2 of the write-up is a *coverage* benefit — the agent sees decisions it would not have queried for — not a benefit from arresting decline. A drift-prone fixture has not been tested.
- **Author-constructed head-to-head queries.** The 50-query MemPalace comparison was constructed by the author and skews toward rationale lookups. The 43–4 / 44–5 splits should not be cited without that caveat.
- **Context overhead at scale is a projection, not a measurement.** Per-call overhead has not been measured on projects with thousands of decisions.
- **Convergence-time runner not executed for v6.4.** Platnick et al.'s 19% / 58% efficiency numbers come from a runner we did not run on this fixture.
- **No user study.** All metrics are automated; we have not measured developer perception.

---

## What's NOT in this repo

- The Continuity VS Code extension itself — see [hackerware.continuity-ultimate on the VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=hackerware.continuity-ultimate).
- The `AutoRetrievalMiddleware` implementation. The `continuity-in-loop` runner condition is a faithful behavioral simulation (re-fire retrieval with augmented context). A reference sketch of the pattern is in Appendix C of the write-up.
- Any proprietary decision data from real projects. Only the sanitized fictional `paydash-api` fixture is here.
- The `head-to-head.ts` MemPalace comparison runner — depends on `@continuity/core`'s `SemanticSearchService`. The summary includes the head-to-head numbers from the broader experiment for completeness; raw artifacts may be vendored in a follow-up release. Implementation questions: contact@hackerware.com.

---

## Citation

```bibtex
@misc{goncalves2026continuity,
  title  = {Continuity: An In-Loop Retrieval Pattern for Persistent Project Rationale in AI Coding Agents},
  author = {Goncalves, Thiago},
  year   = {2026},
  note   = {Version 6.4},
  url    = {https://github.com/alienfader/continuity-benchmarks}
}
```

And the methodology / pattern paper this work adapts:

```bibtex
@misc{platnick2025idrag,
  title  = {ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents},
  author = {Platnick, Daniel and Bengueddache, Mohammed El Amine and Alirezaie, Marjan and Newman, David Jose and Pentland, Alex 'Sandy' and Rahnama, Hossein},
  year   = {2025},
  eprint = {2509.25299},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2509.25299}
}
```

---

## Contact

Academic inquiries, replication support, benchmark extensions, or implementation questions about the Continuity middleware: **contact@hackerware.com**

## License

MIT. See `LICENSE`.
