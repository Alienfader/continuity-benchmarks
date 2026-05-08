/**
 * MCP client utility for `runners/middleware-replay.ts`.
 *
 * Spawns the production Continuity MCP server as a stdio subprocess,
 * connects via @modelcontextprotocol/sdk, and provides typed wrappers
 * around the two tool calls the replay runner needs:
 *
 *   - `search_decisions(query, limit, mode)` — production retrieval
 *     ranker (SemanticSearchService: BM25 + semantic + tags via RRF
 *     fusion). This is the pathway that backs the MCP-tool-using agent
 *     in real deployments.
 *
 *   - `bash({ command })` and `edit({ file_path })` — fire the
 *     AutoRetrievalMiddleware. Decisions arrive in the tool result's
 *     `_meta.relevantDecisions` field. Note: requires `code-links.json`
 *     in the workspace; on fixtures without it, the middleware no-ops
 *     and `_meta.relevantDecisions` will be undefined.
 *
 * Lifecycle:
 *
 *   const client = await McpClient.spawn({
 *     mcpServerPath: '/path/to/dist/index.js',
 *     workspaceRoot: '/path/to/fixture',
 *   });
 *   const decisions = await client.searchDecisions('Why Kafka over Kinesis?', 5);
 *   await client.close();
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

export interface McpDecision {
  id: string;
  question: string;
  answer: string;
  tags?: string[];
  score?: number;
}

export interface SearchDecisionsResult {
  decisions: McpDecision[];
  raw: unknown;
}

export interface ToolCallWithMetaResult {
  /** Decisions injected by AutoRetrievalMiddleware via `_meta.relevantDecisions`, if any. */
  injectedDecisions: McpDecision[];
  /** Whether the middleware was active and injected anything. */
  middlewareFired: boolean;
  /** The raw tool result (text content + _meta). */
  raw: unknown;
}

export interface SpawnArgs {
  /** Absolute path to the MCP server JS bundle (e.g. `dist/index.js`). */
  mcpServerPath: string;
  /** Workspace root the server treats as the active project. */
  workspaceRoot: string;
  /** Extra env vars to inject into the spawned MCP process. */
  env?: Record<string, string>;
  /** If true, forwards the MCP server's stderr to this process's stderr. */
  inheritStderr?: boolean;
}

export class McpClient {
  private constructor(private readonly client: Client) {}

  static async spawn(args: SpawnArgs): Promise<McpClient> {
    const transport = new StdioClientTransport({
      command: process.execPath, // node binary
      args: [args.mcpServerPath],
      // CRITICAL: cwd must be set to the workspace root. The production MCP
      // server's detectWorkspaceRoot() walks up from process.cwd() FIRST to
      // find a `.continuity/` directory; only if that fails does it fall
      // back to WORKSPACE_ROOT. Without setting cwd here, the server picks
      // up whatever .continuity is closest to the runner process — which is
      // usually the benchmarks repo root, not the fixture. We still pass
      // both env vars for belt-and-suspenders.
      cwd: args.workspaceRoot,
      env: {
        ...(process.env as Record<string, string>),
        ...(args.env ?? {}),
        WORKSPACE_ROOT: args.workspaceRoot,
        CONTINUITY_PROJECT_PATH: args.workspaceRoot,
        // Disable network-going telemetry inside the benchmark
        CONTINUITY_TELEMETRY_DISABLED: '1',
      },
      stderr: args.inheritStderr ? 'inherit' : 'pipe',
    });

    const client = new Client(
      { name: 'continuity-benchmarks-middleware-replay', version: '1.0.0' },
      { capabilities: {} },
    );
    await client.connect(transport);
    return new McpClient(client);
  }

  async searchDecisions(
    query: string,
    limit = 5,
    mode: 'keyword' | 'hybrid' = 'hybrid',
  ): Promise<SearchDecisionsResult> {
    const response = await this.client.callTool({
      name: 'search_decisions',
      arguments: { query, limit, mode, includeCodeContext: false, synthesize: false },
    });
    if (process.env.MCP_CLIENT_DEBUG === '1') {
      console.error('[mcp-client.searchDecisions] raw response:', JSON.stringify(response, null, 2).slice(0, 1500));
    }
    const decisions = parseDecisionsFromSearchResult(response);
    return { decisions, raw: response };
  }

  /**
   * Fire the production AutoRetrievalMiddleware via a `bash` tool call.
   * The middleware extracts file-path tokens from the command and looks
   * them up in `<workspace>/.continuity/code-links.json`.
   *
   * If code-links.json is empty/missing OR no decision is linked to the
   * given paths, `injectedDecisions` is empty and `middlewareFired` is
   * false. This is the expected behavior on the public fixtures, which
   * intentionally ship without code-links.
   */
  async invokeMiddleware(
    paths: string[],
  ): Promise<ToolCallWithMetaResult> {
    if (paths.length === 0) {
      return { injectedDecisions: [], middlewareFired: false, raw: null };
    }
    const command = `cat ${paths.map((p) => JSON.stringify(p)).join(' ')}`;
    const response = await this.client.callTool({
      name: 'bash',
      arguments: { command },
    });
    return parseMetaInjection(response);
  }

  async close(): Promise<void> {
    try {
      await this.client.close();
    } catch {
      // Best-effort teardown.
    }
  }
}

// ────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────

/**
 * Extract the leading `{...}` JSON object from a string that may have
 * trailing text after it. Returns null if no balanced object found.
 * Tracks brace depth and respects string boundaries (so `"}"` inside
 * a JSON string doesn't close the outer object).
 */
function extractLeadingJsonObject(text: string): string | null {
  if (text[0] !== '{') return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (inString) {
      if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return text.slice(0, i + 1);
      }
    }
  }
  return null;
}

function parseDecisionsFromSearchResult(result: unknown): McpDecision[] {
  // The MCP `search_decisions` tool returns multiple content blocks. The
  // first is typically a workspace-context warning ("⚠️ UNLOGGED: …
  // commit(s) have no matching decisions"), the second is a JSON-encoded
  // search response of shape:
  //   { synthesis, query, searchMethod, queryEntities, totalMatches,
  //     results: [{ id, question, answer, tags, score, matchType, ... }] }
  // We scan every text block, pick the first one that JSON-parses with a
  // non-empty `results` array, and return that.
  const r = result as {
    content?: Array<{ type?: string; text?: string }>;
    isError?: boolean;
  };
  if (r.isError || !r.content) return [];
  for (const block of r.content) {
    if (block.type !== 'text' || typeof block.text !== 'string') continue;
    const text = block.text.trim();
    if (!text.startsWith('{')) continue;
    // The MCP server's search_decisions response is JSON followed by trailing
    // prose (a guidance footer). Extract just the leading JSON object via
    // brace-matching so JSON.parse doesn't choke on the trailing text.
    const jsonText = extractLeadingJsonObject(text);
    if (!jsonText) continue;
    try {
      const parsed = JSON.parse(jsonText) as {
        results?: Array<Partial<McpDecision> & { decisionId?: string }>;
        decisions?: Array<Partial<McpDecision> & { decisionId?: string }>;
      };
      const list = parsed.results ?? parsed.decisions ?? [];
      if (process.env.MCP_CLIENT_DEBUG === '1') {
        console.error(
          `[mcp-client.parseSearch] parsed list.length=${list.length}, keys=${Object.keys(parsed).join(',')}`,
        );
      }
      if (list.length === 0) continue;
      return list.map((d) => ({
        id: d.id ?? d.decisionId ?? '',
        question: d.question ?? '',
        answer: d.answer ?? '',
        tags: d.tags ?? [],
        score: d.score,
      }));
    } catch (err) {
      if (process.env.MCP_CLIENT_DEBUG === '1') {
        console.error('[mcp-client.parseSearch] JSON.parse threw:', (err as Error).message);
      }
    }
  }
  return [];
}

function parseMetaInjection(result: unknown): ToolCallWithMetaResult {
  const r = result as {
    content?: Array<{ type?: string; text?: string }>;
    isError?: boolean;
    _meta?: {
      relevantDecisions?: Array<{
        id?: string;
        question?: string;
        answer?: string;
        tags?: string[];
      }>;
    };
  };
  const injected: McpDecision[] = [];
  const meta = r._meta?.relevantDecisions ?? [];
  for (const d of meta) {
    injected.push({
      id: d.id ?? '',
      question: d.question ?? '',
      answer: d.answer ?? '',
      tags: d.tags ?? [],
    });
  }
  return {
    injectedDecisions: injected,
    middlewareFired: injected.length > 0,
    raw: result,
  };
}
