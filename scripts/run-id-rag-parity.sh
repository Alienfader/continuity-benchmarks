#!/usr/bin/env bash
#
# run-id-rag-parity.sh — ID-RAG parity matrix for peer-review credibility.
#
# Matrix (current settings — edit FIXTURES / MODELS / RUNS / RUNNERS to widen):
#   2 fixtures × 2 models × 3 runs × 2 runners = 24 invocations
#   Each invocation sweeps all 3 conditions (baseline / continuity / continuity-in-loop)
#   Estimated total cost ~$15-25; wall time ~1.5-2 hours sequentially.
#
# Outputs land in benchmarks/reports/id-rag-parity/<fixture>/<model>/run-<n>/
# Per-invocation log: benchmarks/reports/id-rag-parity/run.log
#
# Runtime: tsx (installed as devDependency). Source files live in
# verification/shared/id-rag-parallel/runners/ — moved there during a
# repo refactor; the previous benchmarks/src/ path is no longer present.
#
set -uo pipefail   # NOT -e: we want to continue past any single-invocation failure

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Load API keys ───────────────────────────────────────────────────────────
if [ -f .env ]; then
  set -a; source .env; set +a
fi

REPORTS=reports/id-rag-parity
LOG=$REPORTS/run.log
mkdir -p "$REPORTS"
: > "$LOG"

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

FIXTURES=(data-pipeline mobile-app)
MODELS=(gpt-4o claude-sonnet-4-6)
RUNS=(1 2 3)
RUNNERS=(recall-over-time action-alignment)

TOTAL=$(( ${#FIXTURES[@]} * ${#MODELS[@]} * ${#RUNS[@]} * ${#RUNNERS[@]} ))
log "matrix start — $TOTAL invocations planned"

i=0
for fixture in "${FIXTURES[@]}"; do
  for model in "${MODELS[@]}"; do
    for run in "${RUNS[@]}"; do
      for runner in "${RUNNERS[@]}"; do
        i=$((i+1))
        outdir="$REPORTS/$fixture/${model//\//_}/run-$run"
        mkdir -p "$outdir"
        log "[$i/$TOTAL] $runner fixture=$fixture model=$model run=$run"

        npx tsx "runners/${runner}.ts" \
          --fixture "$fixture" \
          --model "$model" \
          --seed "$run" \
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

log "matrix complete — see $REPORTS for per-fixture results"
