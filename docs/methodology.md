# Benchmark methodology

See `reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md` for the v2 matrix
analysis (recommended starting point) and `reports/id-rag-parity-summary.md` for
the original v1 paydash-api headline write-up.

## Protocol overview

### Recall-over-time (multi-session drift benchmark)
- 7 sessions × 20 quiz questions per session.
- Conditions (v2 expanded matrix): `baseline` (no retrieval), `continuity-blanket`
  (single retrieval keyed on a concatenation of all 20 question stems, prepended
  to every session), `continuity-perq-frontloaded` (per-question retrieval
  computed once at session 1, the same 20 blobs re-injected unchanged at every
  session boundary — the M2 timing-ablation comparand), and `continuity-in-loop`
  (per-question retrieval re-fired fresh at every session boundary).
- Between sessions, ~5,000 tokens of off-topic noise are injected to simulate
  unrelated agent work.
- Each predicted answer is scored via cosine similarity against ground truth
  using `all-mpnet-base-v2` embeddings. (Note: this is the *evaluation* embedding
  model, intentionally distinct from the `all-MiniLM-L6-v2` embeddings the
  Continuity reference implementation uses for *retrieval*.)

### Action-alignment (single-prompt benchmark)
- 30 prompts × 3 conditions: `baseline`, `continuity` (full-prompt-keyed
  retrieval), `continuity-in-loop` (entity-keyed retrieval — file paths, file-
  shaped tokens, and capitalized identifiers extracted from the prompt, mirroring
  what the production middleware does when it sees a tool call like
  `edit_file(path="src/auth.ts")`).
- Each proposed action is judge-scored 1–10 by `claude-sonnet-4-6`.
- Inter-judge replication: all action-alignment outputs are re-scored by
  `gemini-2.5-flash` (see `runners/re-judge.py` for paydash, `re-judge-cross-
  corpus.py` for the v2 cross-corpus matrix; combined n=1,620 paired scores).

### Head-to-head vs MemPalace (task-oriented contrast)
- 50 queries comparing Continuity's RRF hybrid retrieval against MemPalace's
  flat vector index over the same project repository.
- Note: this is a task-oriented contrast, not a controlled head-to-head over the
  same artifacts. The two systems ingest different artifacts of the same source
  repo: Continuity indexes structured decision records (Q/A pairs from
  `.continuity/decisions.json`) plus a content+filename index over project
  files; MemPalace ingests raw codebase via its default ChromaDB pipeline
  (~1,177 files mined into vector drawers). MemPalace has no equivalent of the
  structured decision corpus by design.
- Each query judged by a single LLM call (Claude Sonnet) which returns relevance
  scores 0–1 and a winner.

## Headline findings (v2 cross-corpus matrix, May 2026)

- **Retrieval ≫ no retrieval.** Continuity beats baseline by 5+ points on
  action alignment (Cohen's d = 8.94, p < 0.005) and 0.17 cosine units on
  recall (d = 11.38).
- **Per-query keying > blanket retrieval** by d = 5.83 (recall, p = 0.003).
- **Timing does not matter (M2 ablation):** holding retrieval data constant and
  varying only injection timing finds mean Δ = −0.002 cosine units, p ≈ 0.05
  slightly favoring the *frozen* condition. Per-tool-call re-firing buys
  nothing measurable over a one-shot session-start injection of the same data.
- **Inter-judge replication (n=1,620 across two studies):** Spearman ρ ∈ [0.72,
  0.79], Cohen's κ ∈ [0.52, 0.56]. Continuity > baseline preserved in every
  cell under both judges (Sonnet 2.30× lift, Gemini 1.82×).

See the v6.4/v7.5 paper for full protocol, scoring rubrics, and limitations
(small n, seven-session horizon, clean decision corpora, self-judging).
