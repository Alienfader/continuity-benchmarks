/**
 * System adapter contract — for plugging alternative memory systems
 * (RAG pipelines, vector DBs, agent-framework memory layers, custom
 * retrieval engines) into the same fixtures + scoring as the built-in
 * conditions.
 *
 * Adapters live in `systems/<name>/index.ts` and export a default
 * `RetrievalSystem`. Runners pick them up via the `--system=<name>` flag
 * (see `recall-over-time.ts` / `action-alignment.ts`).
 *
 * The `Retriever` interface is the same one the built-in BM25Retriever
 * implements, so any adapter that satisfies it slots into the existing
 * condition machinery without runner changes.
 */

import type { Decision } from './fixtures';
import type { Retriever } from './retrieval';

/**
 * What an adapter must export. The `init` function receives the loaded
 * fixture decisions; it returns a Retriever the runner uses for the
 * `continuity-in-loop` (or any other Continuity-*) condition.
 *
 * `init` is async to allow adapters that need to bootstrap a vector
 * store, embed decisions, spin up a sidecar process, etc.
 */
export interface RetrievalSystem {
  /** Stable identifier — should match the directory name under `systems/`. */
  readonly name: string;

  /** Human-readable description shown in run reports. */
  readonly description?: string;

  /**
   * Initialize the adapter against a fixture's decision corpus.
   * Called once per (fixture, model, run) combination.
   */
  init(decisions: Decision[]): Promise<Retriever> | Retriever;
}

/**
 * Dynamically load a system adapter by name. The runner calls this when
 * `--system=<name>` is passed.
 *
 * Resolution order:
 *   1. `systems/<name>/index.ts` (preferred — TypeScript source)
 *   2. `systems/<name>/index.js` (compiled fallback)
 *   3. `systems/<name>.ts` (single-file adapter)
 *
 * Throws if no adapter is found or if the loaded module has no default
 * export matching the `RetrievalSystem` shape.
 */
export async function loadSystem(name: string, projectRoot: string): Promise<RetrievalSystem> {
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    throw new Error(
      `Invalid --system name "${name}". Use only [a-zA-Z0-9_-] (matches the directory name under systems/).`,
    );
  }

  const path = await import('node:path');
  const fs = await import('node:fs');
  const candidates = [
    path.join(projectRoot, 'systems', name, 'index.ts'),
    path.join(projectRoot, 'systems', name, 'index.js'),
    path.join(projectRoot, 'systems', `${name}.ts`),
  ];

  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) {
    throw new Error(
      `System adapter "${name}" not found.\n` +
        `Looked in:\n${candidates.map((c) => `  - ${c}`).join('\n')}\n\n` +
        `See systems/README.md for the adapter contract.`,
    );
  }

  const mod = await import(found);
  const adapter: unknown = mod.default ?? mod;

  if (
    typeof adapter !== 'object' ||
    adapter === null ||
    typeof (adapter as RetrievalSystem).name !== 'string' ||
    typeof (adapter as RetrievalSystem).init !== 'function'
  ) {
    throw new Error(
      `System adapter "${name}" at ${found} is missing required fields.\n` +
        `Must export a default RetrievalSystem with { name: string, init: (decisions) => Retriever }.\n` +
        `See systems/README.md for examples.`,
    );
  }

  return adapter as RetrievalSystem;
}
