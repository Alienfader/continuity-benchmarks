/**
 * `continuity-mcp-middleware` runner — END-TO-END PRODUCTION-MIDDLEWARE REPLAY
 *
 * Status: SKELETON (not yet run end-to-end).
 *
 * This runner exists to close the gap between the §4.7 white-paper claim and
 * what the public `continuity-in-loop` runner actually exercises. The default
 * `continuity-in-loop` condition (in `runners/recall-over-time.ts`) tests the
 * middleware's RETRIEVAL-KEYING logic (entity extraction → BM25 → top-K) but
 * delivers matched decisions by prompt-prepend. The production middleware
 * instead injects matched decisions into the tool result's
 * `_meta.relevantDecisions` field at the point of tool execution.
 *
 * This runner is intended to test the production delivery shape end-to-end.
 *
 * ── DESIGN ─────────────────────────────────────────────────────────────────
 *
 * High-level loop, per quiz question:
 *
 *   1. Spin up the Continuity MCP server as a stdio subprocess.
 *      (Same binary the VS Code extension and CLI use; ships in the
 *      commercial `continuity-ultimate` workspace at
 *      `packages/mcp-server/dist/index.js`.)
 *
 *   2. Issue an MCP tool call that names the relevant fixture file path —
 *      e.g. `read_file({ path: "src/auth.ts" })`. The middleware fires:
 *      it extracts file paths and entity tokens from the tool args, queries
 *      the decision store, and returns the tool result with
 *      `_meta.relevantDecisions` populated.
 *
 *   3. Extract `_meta.relevantDecisions` from the tool result.
 *
 *   4. Format those decisions as the agent's context block (matching the
 *      production extension's prompt-shape — `## Project decisions
 *      (retrieved from Continuity)`, etc.).
 *
 *   5. Make the agent LLM call (same prompts/conditions as the existing
 *      runners) using that context block.
 *
 *   6. Score against ground truth (same scoring path as
 *      `recall-over-time.ts` and `action-alignment.ts`).
 *
 * Comparing this runner's output against `continuity-in-loop` isolates the
 * delivery-shape variable: SAME retrieval keying, SAME ranking, DIFFERENT
 * delivery (`_meta` injection vs prompt-prepend). The §4.7 timing-ablation
 * conclusions are about what's delivered; this runner closes the loop on
 * how it's delivered.
 *
 * ── DEPENDENCIES ───────────────────────────────────────────────────────────
 *
 * This runner needs the production MCP server binary on disk. The expected
 * path is `${CONTINUITY_MCP_PATH}` (env var) or, if unset, an autodetect
 * heuristic that walks up from cwd looking for a sibling `continuity-
 * ultimate/packages/mcp-server/dist/index.js`. The MCP binary itself is
 * NOT in this repo — it ships in the commercial workspace.
 *
 * To run end-to-end:
 *
 *   export CONTINUITY_MCP_PATH=/path/to/continuity-ultimate/packages/mcp-server/dist/index.js
 *   export ANTHROPIC_API_KEY=...
 *   npx tsx runners/middleware-replay.ts \
 *     --fixture data-pipeline --model claude-sonnet-4-6 --seed 1 \
 *     --conditions continuity-mcp-middleware,continuity-in-loop
 *
 * ── STATUS ─────────────────────────────────────────────────────────────────
 *
 * The MCP-client wiring below is intentionally a stub. Implementing it
 * properly requires the `@modelcontextprotocol/sdk` Client + StdioTransport,
 * a session-init handshake, and tool-call dispatch. The path is documented
 * in the production extension's MCP integration code at
 * `src/services/MCPSetup.ts` and `packages/mcp-server/src/index.ts`.
 *
 * The production-middleware run for the v2 matrix is approximately
 * $30–50 in incremental API spend (24 cells × 3 runs at the same per-cell
 * cost as `continuity-in-loop`) and is tracked under `benchmarks/EVAL_PLAN.md`
 * as a pre-registered v3 protocol expansion.
 *
 * ── HOW THIS FILE IS USED IN THE PAPER ─────────────────────────────────────
 *
 * The §2.1 / §3.1 disclosure paragraphs in the v6.4+ white paper reference
 * THIS file by path as evidence that the gap between "what the public
 * benchmark exercises" and "what the production middleware does" is
 * acknowledged and scaffolded for future closure. Removing or renaming
 * this file will break that reference.
 */

/* eslint-disable @typescript-eslint/no-unused-vars */

import * as fs from 'fs';
import * as path from 'path';
import { Decision, loadFixture } from './shared/fixtures';
import { Condition } from './shared/retrieval';

interface MiddlewareReplayArgs {
  fixturePath: string;
  modelName: string;
  seed: number;
  toolCall: { tool: string; args: Record<string, unknown> };
}

interface MiddlewareReplayResult {
  rawToolResult: unknown;
  injectedDecisions: Decision[];
  /** Wall-clock latency from tool-call dispatch to result receipt. */
  latencyMs: number;
}

/**
 * Spin up the production MCP server, issue a tool call, and capture the
 * `_meta.relevantDecisions` payload that the middleware injected.
 *
 * NOT YET IMPLEMENTED. See `DESIGN` and `STATUS` in the file header.
 */
async function replayThroughMiddleware(
  args: MiddlewareReplayArgs,
): Promise<MiddlewareReplayResult> {
  const mcpPath = process.env.CONTINUITY_MCP_PATH;
  if (!mcpPath) {
    throw new Error(
      [
        'middleware-replay: CONTINUITY_MCP_PATH is not set.',
        '',
        'This runner requires the production MCP server binary, which is shipped',
        'as part of the commercial continuity-ultimate workspace. Set:',
        '',
        '  export CONTINUITY_MCP_PATH=/path/to/continuity-ultimate/packages/mcp-server/dist/index.js',
        '',
        'See the file header for full setup. The runner is currently a SKELETON;',
        'see the STATUS section of runners/middleware-replay.ts for implementation',
        'progress.',
      ].join('\n'),
    );
  }

  if (!fs.existsSync(mcpPath)) {
    throw new Error(
      `middleware-replay: CONTINUITY_MCP_PATH points to a nonexistent file: ${mcpPath}`,
    );
  }

  // TODO(continuity-mcp-middleware-runner):
  //   1. Spawn `node ${mcpPath}` as stdio subprocess.
  //   2. Send MCP `initialize` handshake; verify protocol version compat.
  //   3. Send `tools/call` request with `args.toolCall`.
  //   4. Read response; pull `result._meta.relevantDecisions` if present.
  //   5. Map decisions back to fixture format (Decision interface in shared/fixtures).
  //   6. Tear down subprocess; return result.
  //
  // Reference impl path: see `src/services/MCPSetup.ts` and
  // `packages/mcp-server/src/index.ts` in the commercial workspace for how
  // the production extension wires its MCP client up against the same binary.

  throw new Error(
    [
      'middleware-replay: end-to-end MCP client is not yet implemented.',
      '',
      'See the TODO list at the bottom of replayThroughMiddleware() for the',
      'remaining work, and the file header DESIGN section for the architectural',
      'sketch. Tracked under benchmarks/EVAL_PLAN.md as the v3 protocol expansion.',
    ].join('\n'),
  );
}

/**
 * Top-level CLI entry. Mirrors the existing runners' shape so that
 * `npm run bench:matrix-v2` can pick this condition up once it's
 * end-to-end functional.
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.includes('-h') || args.length === 0) {
    console.log(
      [
        'middleware-replay (continuity-mcp-middleware condition) — SKELETON',
        '',
        'See runners/middleware-replay.ts file header for design + status.',
        '',
        'This runner is published as a documented placeholder. End-to-end',
        'implementation is tracked in benchmarks/EVAL_PLAN.md.',
        '',
        'For the §4.7 results the white paper currently reports, see the',
        '`continuity-in-loop` condition wired into runners/recall-over-time.ts',
        'and runners/action-alignment.ts. Those tests exercise the middleware\'s',
        'retrieval-keying logic (entity extraction → BM25 → top-K) and deliver',
        'matched decisions by prompt-prepend. The production delivery shape',
        '(_meta injection at the point of tool execution) is what this runner',
        'will eventually exercise.',
      ].join('\n'),
    );
    process.exit(0);
  }

  console.error('middleware-replay: not yet implemented end-to-end.');
  console.error('Run with --help for status.');
  process.exit(2);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

export { replayThroughMiddleware };
export type { MiddlewareReplayArgs, MiddlewareReplayResult };
