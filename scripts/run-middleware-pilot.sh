#!/bin/bash
#
# Middleware-replay pilot — Tier A scope.
#
# Runs 5 conditions on a single fixture × model × seed cell:
#   1. baseline                (recall-over-time.ts, no retrieval)
#   2. continuity-in-loop      (recall-over-time.ts, simulator: entity extract + BM25)
#   3. mcp-search              (middleware-replay.ts, single-shot via production search_decisions)
#   4. agent-loop              (middleware-replay.ts, 2-turn with search_decisions tool)
#   5. auto-middleware         (middleware-replay.ts, 2-turn with bash → AutoRetrievalMiddleware)
#
# Estimated cost on gpt-4o-mini: ~$0.15 total (5 conditions × 20 questions × ~$0.001-0.002 each).
# Estimated wall-clock: ~10-15 min, dominated by openai latency.
#
# Required env:
#   CONTINUITY_MCP_PATH — absolute path to packages/mcp-server/dist/index.js
#                         in your continuity-ultimate clone.
#   OPENAI_API_KEY      — for gpt-4o-mini agent calls.
#
# Optional:
#   FIXTURE  (default: data-pipeline — the only fixture with code-links.json shipped)
#   MODEL    (default: gpt-4o-mini)
#   SEED     (default: 1)
#   OUT_BASE (default: reports/middleware-pilot/<timestamp>)
#   SESSIONS (default: 1 — recall-over-time can run multi-session; pilot uses 1 to
#             pair the same 20 questions across all five conditions)
#
# After running, scripts/analyze-middleware-pilot.py joins all five results and
# emits per-condition mean cosine + lift over baseline + Cohen's d, plus
# a paired Wilcoxon comparing auto-middleware to continuity-in-loop (the
# headline question: does the production delivery shape preserve the lift?).

set -euo pipefail

FIXTURE="${FIXTURE:-data-pipeline}"
MODEL="${MODEL:-gpt-4o-mini}"
SEED="${SEED:-1}"
SESSIONS="${SESSIONS:-1}"
OUT_BASE="${OUT_BASE:-reports/middleware-pilot/$(date -u +%Y%m%dT%H%M%S)}"

if [[ -z "${CONTINUITY_MCP_PATH:-}" ]]; then
  echo "ERROR: CONTINUITY_MCP_PATH must be set." >&2
  echo "  Example: export CONTINUITY_MCP_PATH=/path/to/packages/mcp-server/dist/index.js" >&2
  exit 1
fi
if [[ ! -f "${CONTINUITY_MCP_PATH}" ]]; then
  echo "ERROR: CONTINUITY_MCP_PATH points to a nonexistent file: ${CONTINUITY_MCP_PATH}" >&2
  exit 1
fi
if [[ "${MODEL}" != "mock" && -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "WARNING: neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set." >&2
  echo "  This run will fail unless --model=mock." >&2
fi

mkdir -p "${OUT_BASE}"
echo "[pilot] fixture=${FIXTURE} model=${MODEL} seed=${SEED} sessions=${SESSIONS}"
echo "[pilot] output → ${OUT_BASE}"
echo ""

# ── Step 1: recall-over-time (baseline + continuity-in-loop) ────────────────
echo "[pilot] step 1/4: recall-over-time (baseline + continuity-in-loop)"
mkdir -p "${OUT_BASE}/recall-over-time"
npx tsx runners/recall-over-time.ts \
  --fixture "${FIXTURE}" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --sessions "${SESSIONS}" \
  --conditions "baseline,continuity-in-loop" \
  --output "${OUT_BASE}/recall-over-time/result"
echo ""

# ── Steps 2-4: middleware-replay × 3 modes ──────────────────────────────────
for MODE in mcp-search agent-loop auto-middleware; do
  echo "[pilot] step (mode=${MODE})"
  mkdir -p "${OUT_BASE}/middleware-replay-${MODE}"
  npx tsx runners/middleware-replay.ts \
    --fixture "${FIXTURE}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --retrieval="${MODE}" \
    --output "${OUT_BASE}/middleware-replay-${MODE}"
  echo ""
done

# ── Analysis ────────────────────────────────────────────────────────────────
echo "[pilot] analysis"
python3 scripts/analyze-middleware-pilot.py "${OUT_BASE}"
echo ""
echo "[pilot] complete. Reports → ${OUT_BASE}/"
