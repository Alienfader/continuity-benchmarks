#!/usr/bin/env bash
#
# run-id-rag-parity-v2-resume.sh — targeted re-run of the 12 cells that
# failed during the initial v2 matrix run on 2026-05-06 due to Anthropic
# credit exhaustion (11 cells) and a transient OpenAI 502 (1 cell).
#
# Idempotent: skips any cell whose output JSON already exists, so this
# script is safe to re-invoke if credits run out again partway through.
#
# Estimated cost: ~$10-15 (mostly claude-sonnet-4-6 cells; mobile-app is
# both fixture's harder corpus and 9 of the 12 missing cells use claude
# either as agent or as judge). Wall time: ~5-6h sequentially.
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

REPORTS=reports/id-rag-parity-v2
LOG=$REPORTS/resume.log
mkdir -p "$REPORTS"
: > "$LOG"

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

RECALL_CONDITIONS="baseline,continuity-blanket,continuity-perq-frontloaded,continuity-in-loop"
ALIGNMENT_CONDITIONS="baseline,continuity,continuity-in-loop"

# Explicit list of (fixture, model, run, runner) tuples to re-run.
# Format: fixture|model|run|runner
TARGETS=(
  "data-pipeline|gpt-4o|2|recall-over-time"
  "data-pipeline|claude-sonnet-4-6|3|recall-over-time"
  "data-pipeline|claude-sonnet-4-6|3|action-alignment"
  "mobile-app|gpt-4o|1|action-alignment"
  "mobile-app|gpt-4o|2|action-alignment"
  "mobile-app|gpt-4o|3|action-alignment"
  "mobile-app|claude-sonnet-4-6|1|recall-over-time"
  "mobile-app|claude-sonnet-4-6|1|action-alignment"
  "mobile-app|claude-sonnet-4-6|2|recall-over-time"
  "mobile-app|claude-sonnet-4-6|2|action-alignment"
  "mobile-app|claude-sonnet-4-6|3|recall-over-time"
  "mobile-app|claude-sonnet-4-6|3|action-alignment"
)

TOTAL=${#TARGETS[@]}
log "matrix v2 resume start — $TOTAL targets queued"

i=0
skipped=0
for target in "${TARGETS[@]}"; do
  i=$((i+1))
  IFS='|' read -r fixture model run runner <<< "$target"

  outdir="$REPORTS/$fixture/${model//\//_}/run-$run"
  mkdir -p "$outdir"
  outjson="$outdir/${runner}.json"

  if [ -f "$outjson" ]; then
    log "[$i/$TOTAL] $runner $fixture/$model/run-$run — output exists, skipping"
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
    log "  ! exit=$rc (continuing resume; re-run script to retry)"
  else
    log "  ✓ done"
  fi
done

log "matrix v2 resume complete — skipped=$skipped of $TOTAL"
