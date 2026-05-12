/**
 * Minimal CLI argument parser used by each runner — no commander dependency.
 *
 * Supports:
 *   --flag              (boolean true)
 *   --key value         (string)
 *   --key=value         (string)
 *   --key 1 --key 2     (string[]; last wins unless `repeat: true` on the schema)
 */

import type { SupportedModel } from './llm-providers';

export interface RunnerArgs {
  fixture: string;
  model: SupportedModel;
  conditions: Array<'baseline' | 'continuity' | 'continuity-in-loop'>;
  seed: number;
  mock: boolean;
  output?: string;
  /** Runner-specific: recall-over-time session count. */
  sessions?: number;
  /** Runner-specific: retrieval top-K. */
  topK?: number;
  /** Runner-specific: action-alignment prompt count. */
  actions?: number;
  /** Runner-specific: convergence-time step count. */
  steps?: number;
  /**
   * Custom retrieval-system adapter name (looks up `systems/<name>/index.ts`).
   * When set, the runner uses the adapter's retriever in place of the
   * built-in BM25 for any Continuity-* condition. See systems/README.md.
   */
  system?: string;
  /** Verbose logging. */
  verbose?: boolean;
}

export function parseArgs(argv: string[]): RunnerArgs {
  const raw: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const eqIdx = token.indexOf('=');
    if (eqIdx !== -1) {
      const key = token.slice(2, eqIdx);
      raw[key] = token.slice(eqIdx + 1);
      continue;
    }
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      raw[key] = true;
    } else {
      raw[key] = next;
      i += 1;
    }
  }

  const fixture = pickString(raw, 'fixture') ?? 'paydash-api';
  const model = (pickString(raw, 'model') ?? 'mock') as SupportedModel;
  const conditionsStr = pickString(raw, 'conditions') ?? 'baseline,continuity,continuity-in-loop';
  const conditions = conditionsStr.split(',').map((s) => s.trim()) as RunnerArgs['conditions'];
  const seed = Number(pickString(raw, 'seed') ?? 42);
  const mock = raw.mock === true || model === 'mock';
  const output = pickString(raw, 'output');
  const sessions = maybeNumber(raw, 'sessions');
  const topK = maybeNumber(raw, 'topK') ?? maybeNumber(raw, 'top-k');
  const actions = maybeNumber(raw, 'actions');
  const steps = maybeNumber(raw, 'steps');
  const system = pickString(raw, 'system');
  const verbose = raw.verbose === true;

  return { fixture, model, conditions, seed, mock, output, sessions, topK, actions, steps, system, verbose };
}

function pickString(raw: Record<string, string | boolean>, key: string): string | undefined {
  const v = raw[key];
  return typeof v === 'string' ? v : undefined;
}

function maybeNumber(raw: Record<string, string | boolean>, key: string): number | undefined {
  const v = pickString(raw, key);
  if (v === undefined) return undefined;
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`--${key} must be a number, got "${v}"`);
  return n;
}

export function printHelp(runnerName: string, extraFlags: string[] = []): void {
  const common = [
    '  --fixture <name>        Fixture project (default: paydash-api)',
    '  --model <name>          mock | gpt-4o | gpt-4o-mini | qwen2.5-7b | claude-sonnet-4-6',
    '  --conditions <csv>      Comma-separated (default: baseline,continuity,continuity-in-loop)',
    '  --seed <n>              Deterministic seed (default: 42)',
    '  --mock                  Force mock LLM (no network)',
    '  --output <path>         Override report output base path',
    '  --system <name>         Use a custom retrieval adapter from systems/<name>/ (see systems/README.md)',
    '  --verbose               Print per-question progress',
  ];
  console.log(`Usage: npx ts-node benchmarks/src/id-rag-parallel/runners/${runnerName}.ts [flags]\n`);
  for (const line of [...common, ...extraFlags]) console.log(line);
}
