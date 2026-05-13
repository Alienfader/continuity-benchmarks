# LongMemEval fixture

External benchmark dataset from
[xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(MIT licensed). Used by `runners/longmemeval.ts`.

## Setup

The raw dataset (~280 MB) and any stratified subsamples are **not tracked**
in this repo. Regenerate from the public source:

```bash
# 1. Download the raw _s file (280 MB) from HuggingFace
curl -sSLo fixtures/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 2. Generate the 50-question stratified subsample (deterministic, seed=42)
python3 scripts/longmemeval-subsample.py
```

The subsample script produces `sample-50.json` (~26 MB) covering all 6 question
types: knowledge-update, multi-session, single-session-{assistant,user,preference},
temporal-reasoning. Default is 8-9 questions per type.

## Running the benchmark

See `runners/longmemeval.ts` header for the full usage. Quick start:

```bash
GOOGLE_API_KEY=... npx tsx runners/longmemeval.ts \
  --sample fixtures/longmemeval/sample-50.json \
  --output reports/longmemeval/run-1
```

Default model is Gemini 2.5 Flash for both agent and judge — cheap (~$0.10
total for the 50-question run) and within the free-tier budget for most users.

## Caveats (read before citing results)

1. **Baseline = no context**, not the full 115k-token chat history. This
   means continuity-vs-baseline measures "does retrieval help?", *not*
   "does retrieval beat a long-context model with the full history?".
   To compare against the official LongMemEval leaderboard, you need the
   115k-context baseline and the official GPT-4o judge.
2. **Judge is Gemini Flash**, not the official GPT-4o judge. Re-judging
   a subsample with GPT-4o is the path to leaderboard-comparable numbers.
3. **Subsample, not the full 500.** The seed-stratified 50 preserves the
   six question-type proportions but is not statistically equivalent to
   a full run.
