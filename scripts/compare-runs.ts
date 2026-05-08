/**
 * compare-runs.ts — Reads two recall-over-time / action-alignment JSONs
 * and emits a delta summary. Used by bench:compare and the GitHub
 * Actions PR-comment workflow.
 *
 * Usage:
 *   ts-node scripts/compare-runs.ts \
 *     --baseline=reports/baseline/recall-over-time.json \
 *     --custom=reports/my-run/recall-over-time.json \
 *     --output=reports/summary.json
 *
 * Output JSON shape (self-describing — comparison-bot prints it as-is):
 *   {
 *     baseline:    { path, condition->meanScore },
 *     custom:      { path, condition->meanScore },
 *     deltas:      { condition: customMean - baselineMean }
 *     winner:      'baseline' | 'custom' | 'tie',
 *     summaryText: "1-line headline"
 *   }
 *
 * Works on either runner type — recognizes both shapes by `benchmark` key.
 */

import * as fs from 'fs';
import * as path from 'path';

interface Args {
  baseline: string;
  custom: string;
  output?: string;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const out: Partial<Args> = {};
  for (const tok of argv) {
    const eq = tok.indexOf('=');
    if (!tok.startsWith('--') || eq === -1) continue;
    const key = tok.slice(2, eq);
    const val = tok.slice(eq + 1);
    if (key === 'baseline' || key === 'custom' || key === 'output') {
      out[key] = val;
    }
  }
  if (!out.baseline || !out.custom) {
    console.error('Usage: compare-runs --baseline=<path> --custom=<path> [--output=<path>]');
    process.exit(1);
  }
  return out as Args;
}

function readReport(p: string): { conditions: Record<string, number>; benchmark: string; system?: string } {
  if (!fs.existsSync(p)) {
    throw new Error(`Report not found: ${p}`);
  }
  const data = JSON.parse(fs.readFileSync(p, 'utf8'));
  if (data.benchmark === 'recall-over-time') {
    const conditions: Record<string, number> = {};
    for (const c of data.conditions ?? []) {
      conditions[c.condition] = c.meanAcrossSessions;
    }
    return { benchmark: data.benchmark, conditions, system: data.system };
  }
  if (data.benchmark === 'action-alignment') {
    const conditions: Record<string, number> = {};
    for (const s of data.summaries ?? []) {
      conditions[s.condition] = s.meanScore;
    }
    return { benchmark: data.benchmark, conditions, system: data.system };
  }
  throw new Error(`Unknown benchmark type in ${p}: ${data.benchmark}`);
}

function main() {
  const args = parseArgs();
  const base = readReport(args.baseline);
  const cust = readReport(args.custom);

  if (base.benchmark !== cust.benchmark) {
    throw new Error(
      `Cannot compare different benchmarks: baseline=${base.benchmark}, custom=${cust.benchmark}`,
    );
  }

  const allConditions = new Set([...Object.keys(base.conditions), ...Object.keys(cust.conditions)]);
  const deltas: Record<string, number | null> = {};
  for (const cond of allConditions) {
    const b = base.conditions[cond];
    const c = cust.conditions[cond];
    if (b === undefined || c === undefined) {
      deltas[cond] = null;
    } else {
      deltas[cond] = Number((c - b).toFixed(4));
    }
  }

  // Headline pick: largest absolute delta across shared conditions.
  const sharedDeltas = Object.entries(deltas).filter(([, d]) => d !== null) as Array<[string, number]>;
  let headlineCondition: string | null = null;
  let headlineDelta = 0;
  for (const [cond, d] of sharedDeltas) {
    if (Math.abs(d) >= Math.abs(headlineDelta)) {
      headlineDelta = d;
      headlineCondition = cond;
    }
  }

  let winner: 'baseline' | 'custom' | 'tie' = 'tie';
  if (headlineDelta > 0.001) winner = 'custom';
  else if (headlineDelta < -0.001) winner = 'baseline';

  let summaryText: string;
  if (sharedDeltas.length === 0) {
    summaryText = 'No shared conditions between baseline and custom runs.';
  } else if (headlineCondition === null) {
    summaryText = `${cust.benchmark}: tie across all ${sharedDeltas.length} shared conditions.`;
  } else {
    const arrow = winner === 'custom' ? '↑' : winner === 'baseline' ? '↓' : '≈';
    summaryText = `${cust.benchmark}: custom ${arrow} on ${headlineCondition} (Δ ${headlineDelta >= 0 ? '+' : ''}${headlineDelta.toFixed(4)})`;
  }

  const summary = {
    benchmark: base.benchmark,
    baseline: { path: args.baseline, system: base.system ?? 'built-in', conditions: base.conditions },
    custom: { path: args.custom, system: cust.system ?? 'built-in', conditions: cust.conditions },
    deltas,
    winner,
    summaryText,
    generatedAt: new Date().toISOString(),
  };

  const json = JSON.stringify(summary, null, 2);
  console.log(json);

  if (args.output) {
    fs.mkdirSync(path.dirname(args.output), { recursive: true });
    fs.writeFileSync(args.output, json + '\n', 'utf8');
    console.error(`[compare-runs] wrote ${args.output}`);
  }
}

main();
