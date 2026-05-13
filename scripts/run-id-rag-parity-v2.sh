#!/usr/bin/env bash
#
# run-id-rag-parity-v2.sh — ID-RAG parity matrix v2 with corrected runners.
#
# Differences from run-id-rag-parity.sh (v1, v6.3 paper):
#
#   1. Output goes to benchmarks/reports/id-rag-parity-v2/ — v6.3 data
#      under benchmarks/reports/id-rag-parity/ is preserved untouched.
#
#   2. action-alignment runs the corrected `continuity-in-loop` condition
#      (entity-keyed retrieval, mirroring AutoRetrievalMiddleware's file-
#      path extraction). v6.3's runner had `continuity` and `in-loop` as
#      identical code paths — this is the fix.
#
#   3. recall-over-time runs the 4-condition expanded matrix:
#        - baseline
#        - continuity-blanket            (= legacy `continuity`)
#        - continuity-perq-frontloaded   (M2 ablation comparand)
#        - continuity-in-loop            (per-question fresh re-retrieval)
#
#      Adds the per-question-frontloaded condition that holds retrieval
#      data constant while varying injection timing. Pairwise contrasts:
#        baseline → blanket           : effect of any retrieval
#        blanket → perq-frontloaded   : effect of better keying (timing held)
#        perq-frontloaded → in-loop   : effect of fresh re-retrieval (timing only)
#
#   4. Cost estimate: ~$25-30 (vs v1's $20). The 4th recall condition adds
#      ~33% to recall LLM calls; action-alignment cost unchanged.
#
#   5. Wall time: ~14h sequentially.
#
# Overwrite guard: cells whose output JSON already exists are skipped by
# default to protect prior 14h-matrix data. To re-run a populated matrix
# from scratch, invoke with `FORCE=1 scripts/run-id-rag-parity-v2.sh` or
# delete the target cells under reports/id-rag-parity-v2/ first.
#
# Runtime: tsx (installed as devDependency).
#
set -uo pipefail   # NOT -e: continue past any single-invocation failure

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Load API keys ───────────────────────────────────────────────────────────
if [ -f .env ]; then
  set -a; source .env; set +a
fi

REPORTS=reports/id-rag-parity-v2
LOG=$REPORTS/run.log
mkdir -p "$REPORTS"
: > "$LOG"

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

FIXTURES=(data-pipeline mobile-app)
MODELS=(gpt-4o claude-sonnet-4-6)
RUNS=(1 2 3)
RUNNERS=(recall-over-time action-alignment)

# Per-runner condition strings (passed via --conditions)
RECALL_CONDITIONS="baseline,continuity-blanket,continuity-perq-frontloaded,continuity-in-loop"
ALIGNMENT_CONDITIONS="baseline,continuity,continuity-in-loop"

TOTAL=$(( ${#FIXTURES[@]} * ${#MODELS[@]} * ${#RUNS[@]} * ${#RUNNERS[@]} ))
log "matrix v2 start — $TOTAL invocations planned"
log "  recall conditions:    $RECALL_CONDITIONS"
log "  alignment conditions: $ALIGNMENT_CONDITIONS"

FORCE="${FORCE:-0}"
if [ "$FORCE" = "1" ]; then
  log "FORCE=1 set — existing cells WILL be overwritten"
else
  log "overwrite guard active — existing cells will be skipped (set FORCE=1 to re-run)"
fi

i=0
skipped=0
for fixture in "${FIXTURES[@]}"; do
  for model in "${MODELS[@]}"; do
    for run in "${RUNS[@]}"; do
      for runner in "${RUNNERS[@]}"; do
        i=$((i+1))
        outdir="$REPORTS/$fixture/${model//\//_}/run-$run"
        mkdir -p "$outdir"
        outjson="$outdir/${runner}.json"

        if [ -f "$outjson" ] && [ "$FORCE" != "1" ]; then
          log "[$i/$TOTAL] $runner $fixture/$model/run-$run — output exists, skipping (set FORCE=1 to overwrite)"
          skipped=$((skipped+1))
          continue
        fi

        if [ "$runner" = "recall-over-time" ]; then
          conditions="$RECALL_CONDITIONS"
        else
          conditions="$ALIGNMENT_CONDITIONS"
        fi

        log "[$i/$TOTAL] $runner fixture=$fixture model=$model run=$run conditions=$conditions"

        npx tsx "runners/${runner}.ts" \
          --fixture "$fixture" \
          --model "$model" \
          --seed "$run" \
          --conditions "$conditions" \
          --output "$outdir/${runner}" \
          >> "$LOG" 2>&1

        rc=$?
        if [ $rc -ne 0 ]; then
          log "  ! exit=$rc (continuing matrix)"
        else
          log "  ✓ done"
        fi
      done
    done
  done
done

log "matrix v2 complete — skipped=$skipped of $TOTAL — see $REPORTS for per-fixture results"
