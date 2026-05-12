#!/usr/bin/env bash
# bench-custom.sh — Run a built-in runner with a custom retrieval-system
# adapter loaded from systems/<name>/index.ts.
#
# Usage:
#   ./scripts/bench-custom.sh \
#     --system=my-vector-db \
#     --runner=recall|alignment \
#     --fixture=paydash-api \
#     --model=gpt-4o-mini \
#     --seed=1 \
#     --output=reports/my-run/recall
#
# The --runner flag picks the underlying scoring lane:
#   - recall     → runners/recall-over-time.ts (multi-session drift)
#   - alignment  → runners/action-alignment.ts (LLM-judge 1-10)
#
# All other flags pass through to the runner unchanged. See
# systems/README.md for the adapter contract.

set -euo pipefail

RUNNER=""
FORWARD=()

for arg in "$@"; do
  case "$arg" in
    --runner=*)
      RUNNER="${arg#*=}"
      ;;
    --runner)
      echo "error: --runner requires =VALUE form (e.g. --runner=recall)" >&2
      exit 1
      ;;
    *)
      FORWARD+=("$arg")
      ;;
  esac
done

if [ -z "$RUNNER" ]; then
  echo "error: --runner=<recall|alignment> is required" >&2
  echo "usage: bench:custom -- --system=<name> --runner=<recall|alignment> [other flags]" >&2
  exit 1
fi

case "$RUNNER" in
  recall|recall-over-time)
    SCRIPT="runners/recall-over-time.ts"
    ;;
  alignment|action-alignment)
    SCRIPT="runners/action-alignment.ts"
    ;;
  *)
    echo "error: unknown --runner=\"$RUNNER\". Must be 'recall' or 'alignment'." >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."
exec npx ts-node --transpile-only "$SCRIPT" "${FORWARD[@]}"
