# Reproducing the benchmarks

## Prerequisites
- Node 18+ and npm
- Python 3.10+ (for the re-judge.py / re-judge-cross-corpus.py scripts)
- API keys in a `.env` file at the repo root:
  - `ANTHROPIC_API_KEY` (primary LLM judge; also used if benchmarking Claude
    Sonnet 4.6 as agent)
  - `OPENAI_API_KEY` (for GPT-4o agent — needs paid tier; Tier 1 hits 3 RPM
    limits on GPT-4o)
  - `GOOGLE_API_KEY` (for Gemini inter-judge cross-validation — needs billing
    enabled at ai.studio/projects)

## Single benchmark invocation
```bash
npm install
npx tsx runners/recall-over-time.ts \
  --fixture paydash-api \
  --model gpt-4o-mini \
  --seed 1
```

## Full v2 cross-corpus matrix (~12-14h wall clock, ~$25-30 at Tier 2)
```bash
bash scripts/run-id-rag-parity-v2.sh
```
- 24 invocations: 2 fixtures × 2 agent models × 3 runs × 2 runners.
- Recall runs all 4 conditions (baseline, blanket, perq-frontloaded, in-loop).
- Action-alignment runs 3 conditions (baseline, continuity, in-loop).
- Outputs land in `reports/id-rag-parity-v2/`.
- If credit exhaustion or transient API failures interrupt the run, resume
  with `bash scripts/run-id-rag-parity-v2-resume.sh` (idempotent, skips cells
  whose output JSON already exists).

## Cross-validating the judge
```bash
# v1 paydash inter-judge (n=540, Sonnet vs Gemini-2.5-flash)
python3 runners/re-judge.py

# v2 cross-corpus inter-judge (n=1,080, Sonnet vs Gemini-2.5-flash)
python3 runners/re-judge-cross-corpus.py
```
Both scripts emit Cohen's κ + Spearman ρ against Sonnet's original scores and
write paired-score JSON to `reports/id-rag-parity/`.

## Replicating the v2 analysis (no API spend, ~5 seconds)
```bash
python3 runners/experimental-gaps-analysis-v2.py
```
Reads the v2 matrix outputs and emits paired Wilcoxon + Cohen's d for the M2
timing ablation contrast plus reference contrasts. Writes
`reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md` and the JSON.
