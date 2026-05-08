/**
 * Production-middleware replay runner. Three retrieval modes, all
 * end-to-end through real MCP stdio transport:
 *
 *   1. mcp-search     (default) — single-shot. Calls the production
 *                                 search_decisions MCP tool with the
 *                                 prompt as the query. Tests the
 *                                 production retrieval ranker
 *                                 (SemanticSearchService RRF hybrid)
 *                                 delivered via MCP. No agent reasoning.
 *
 *   2. agent-loop                — 2-turn loop. The agent has
 *                                 search_decisions advertised; it
 *                                 decides whether/how to query, MCP
 *                                 returns real decisions through the
 *                                 production ranker, the agent
 *                                 generates the final answer in turn 2.
 *                                 Tests the full agent + production
 *                                 retrieval ranker via real MCP.
 *
 *   3. auto-middleware           — 2-turn loop. The agent has the
 *                                 benchmark-mode `bash` tool advertised;
 *                                 it issues `cat <path>` calls. The
 *                                 production AutoRetrievalMiddleware
 *                                 fires on path-bearing tool args, looks
 *                                 up code-links.json, and injects
 *                                 matched decisions into
 *                                 _meta.relevantDecisions. Tests the
 *                                 production AutoRetrievalMiddleware
 *                                 delivery shape end-to-end.
 *
 *                                 Requires:
 *                                 (a) MCP server with
 *                                     CONTINUITY_BENCHMARK_MODE=1, which
 *                                     exposes the read-only `bash` tool.
 *                                 (b) code-links.json in the workspace
 *                                     mapping decision IDs to file
 *                                     paths the agent can reference.
 *                                 The data-pipeline fixture ships with
 *                                 both pre-configured. The runner
 *                                 forwards CONTINUITY_BENCHMARK_MODE=1
 *                                 to the spawned server automatically
 *                                 when this mode is selected.
 *
 * Smoke tests pass for all three modes (mock model). The v2 matrix run
 * for any mode is gated on user green-light because it costs ~$30–50
 * in API spend.
 *
 * ── WHY THIS RUNNER EXISTS ────────────────────────────────────────────────
 *
 * The default `continuity-in-loop` runner (in `runners/recall-over-time.ts`)
 * tests the production middleware's RETRIEVAL-KEYING logic (entity extract
 * → BM25 → top-K) using a self-contained simulator. It does NOT exercise:
 *
 *   1. The production retrieval ranker (`SemanticSearchService`: BM25 +
 *      semantic embeddings + tags fused via reciprocal rank fusion). The
 *      simulator uses only naive BM25 over decision Q+A+tags.
 *   2. The MCP transport layer that delivers retrieval to the agent in
 *      production (decisions arrive in tool result `_meta.relevantDecisions`
 *      rather than prepended to the system prompt).
 *
 * This runner closes both gaps by running through the real production MCP
 * server. It supports two retrieval modes:
 *
 *   --retrieval=mcp-search   (DEFAULT)
 *     Calls the MCP `search_decisions` tool with the prompt as the query.
 *     Server-side this hits the production SemanticSearchService → real
 *     RRF hybrid retrieval. Top-K decisions are formatted into the agent
 *     prompt the same way the in-loop runner formats its retrieval. This
 *     mode tests the production retrieval ranker delivered via MCP, which
 *     is what most agent operators actually get when they use Continuity.
 *
 *   --retrieval=auto-middleware
 *     Issues a `bash` tool call referencing the path-shaped tokens
 *     extracted from the prompt; the production AutoRetrievalMiddleware
 *     fires server-side, looks up `<workspace>/.continuity/code-links.json`,
 *     and injects matched decisions into `_meta.relevantDecisions`. This
 *     tests the file-path-keyed delivery shape end-to-end.
 *
 *     ⚠️ PRECONDITION: the workspace must have `code-links.json`. The
 *     public fixtures intentionally ship without one (their decisions
 *     reference systems by name like "Kafka" / "MSK", not file paths).
 *     On those fixtures this mode will return zero decisions per call —
 *     which is the honest production-replay result given the fixture
 *     shape, and matches what production deployments without explicit
 *     code-linking would deliver.
 *
 * ── HOW THIS RUNNER COMPARES TO `continuity-in-loop` ──────────────────────
 *
 * | Variable           | continuity-in-loop (sim) | mcp-search       | auto-middleware       |
 * |--------------------|--------------------------|------------------|-----------------------|
 * | Retrieval ranker   | naive BM25 (in-process)  | RRF hybrid (prod)| code-links lookup     |
 * | Delivery           | prompt-prepend           | prompt-prepend   | tool-result `_meta`   |
 * | MCP transport      | not used                 | real             | real                  |
 * | Code-links needed  | no                       | no               | yes                   |
 *
 * The contrast `continuity-in-loop vs mcp-search` isolates the retrieval-
 * ranker variable. The contrast `mcp-search vs auto-middleware` (for a
 * code-links-bearing workspace) isolates the delivery-shape variable. The
 * §4.7 timing-ablation conclusion ("retrieval specificity dominates timing")
 * survives if BOTH contrasts produce non-zero recall lifts; it would be
 * walked back further if either degrades the result.
 *
 * ── DEPENDENCIES ──────────────────────────────────────────────────────────
 *
 * Required env vars:
 *   - CONTINUITY_MCP_PATH     — absolute path to the production MCP
 *                               server bundle (e.g.
 *                               .../packages/mcp-server/dist/index.js)
 *   - ANTHROPIC_API_KEY       — for the agent + judge LLM calls (if
 *                               --model claude-* is used)
 *   - OPENAI_API_KEY          — for the agent LLM calls (if
 *                               --model gpt-* is used)
 *
 * The MCP server bundle is NOT shipped in this repo — it lives in the
 * commercial `continuity-ultimate` workspace. To run end-to-end, point
 * CONTINUITY_MCP_PATH at a local clone.
 *
 * ── INVOCATION ────────────────────────────────────────────────────────────
 *
 *   # Smoke test (mock model, no API spend, ~30 sec):
 *   export CONTINUITY_MCP_PATH=/path/to/packages/mcp-server/dist/index.js
 *   npx tsx runners/middleware-replay.ts \
 *     --fixture data-pipeline --model mock --questions 3 \
 *     --retrieval=mcp-search --output /tmp/smoke-mcp-replay
 *
 *   # Single real-model invocation (~$0.20):
 *   npx tsx runners/middleware-replay.ts \
 *     --fixture data-pipeline --model gpt-4o-mini --questions 20 \
 *     --retrieval=mcp-search --seed 1 --output reports/mcp-replay-run-1
 *
 * ── STATUS ────────────────────────────────────────────────────────────────
 *
 * 2026-05-08 — initial implementation. End-to-end smoke tests pass on
 * the data-pipeline fixture in both retrieval modes (mock model). The
 * `mcp-search` mode retrieves real decisions from the production MCP
 * server's search_decisions tool (verified retrieved=1..5 per question
 * across the 20-question quiz). The `auto-middleware` mode correctly
 * shows fire-rate=0% on fixtures without `code-links.json`, which is
 * the honest production-replay result given the fixture shape. Full v2
 * matrix run (24 cells × 3 runs × 2 modes) is pending user green-light;
 * estimated $30–50 in API spend.
 */

/* eslint-disable @typescript-eslint/no-unused-vars */

import * as fs from 'fs';
import * as path from 'path';
import { McpClient, McpDecision } from './shared/mcp-client';
import { LLMClient, createLLMClient, SupportedModel } from './shared/llm-providers';
import { Embedder, EvalEmbedder, MockEmbedder, cosineSimilarity } from './shared/eval-embeddings';
import {
  ToolCallingAgent,
  ToolDef,
  ToolResult,
  createAgent,
} from './shared/agent-client';

type RetrievalMode = 'mcp-search' | 'auto-middleware' | 'agent-loop';

interface ReplayArgs {
  fixtureName: string;
  /** Optional: absolute path to a workspace root (overrides fixtures/<name>/). */
  workspaceOverride?: string;
  /** Optional: absolute path to a quiz JSON file (overrides prompts/quizzes/<name>.json). */
  quizOverride?: string;
  modelName: string;
  retrievalMode: RetrievalMode;
  questionsPerSession: number;
  topK: number;
  seed: number;
  outputDir: string;
  verbose: boolean;
}

interface PerQuestionResult {
  questionId: string;
  question: string;
  retrievedDecisionIds: string[];
  middlewareFired: boolean;
  agentAnswer: string;
  cosineVsGroundTruth: number;
  inputTokens: number;
  outputTokens: number;
}

interface RunReport {
  fixture: string;
  model: string;
  retrievalMode: RetrievalMode;
  topK: number;
  seed: number;
  questions: PerQuestionResult[];
  meanCosine: number;
  fractionAbove070: number;
  middlewareFireRate: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  durationMs: number;
}

function parseArgs(argv: string[]): ReplayArgs {
  const get = (flag: string, fallback?: string): string | undefined => {
    const idx = argv.indexOf(flag);
    if (idx >= 0 && idx + 1 < argv.length) return argv[idx + 1];
    const eq = argv.find((a) => a.startsWith(`${flag}=`));
    if (eq) return eq.slice(flag.length + 1);
    return fallback;
  };

  const has = (flag: string): boolean => argv.includes(flag);

  if (has('--help') || has('-h')) {
    console.log(
      [
        'middleware-replay — production-middleware replay runner (continuity-mcp-middleware condition)',
        '',
        'Usage:',
        '  npx tsx runners/middleware-replay.ts \\',
        '    --fixture <name> --model <name> --retrieval=<mode> \\',
        '    [--questions N] [--top-k K] [--seed S] [--output DIR] [--verbose]',
        '',
        'Args:',
        '  --fixture     Fixture name from prompts/quizzes/ (e.g. data-pipeline)',
        '  --model       LLM identifier (mock, gpt-4o, gpt-4o-mini, claude-sonnet-4-6)',
        '  --retrieval   one of:',
        '                  "mcp-search"     — calls search_decisions on the prompt directly',
        '                                     (single-shot, no agent reasoning)',
        '                  "agent-loop"     — 2-turn loop, agent decides whether/how to',
        '                                     query search_decisions; tests full agent +',
        '                                     production retrieval ranker via real MCP',
        '                  "auto-middleware" — 2-turn loop with bash tool advertised; tests',
        '                                     AutoRetrievalMiddleware delivery via _meta;',
        '                                     requires CONTINUITY_BENCHMARK_MODE=1 server build',
        '                                     and code-links.json in workspace',
        '  --questions   Questions per run (default: full quiz)',
        '  --top-k       Decisions retrieved per question (default: 5)',
        '  --seed        RNG seed for repro (default: 1)',
        '  --output      Output directory (default: /tmp/middleware-replay-<ts>)',
        '  --verbose     Print per-question scoring',
        '',
        'Required env:',
        '  CONTINUITY_MCP_PATH    Path to packages/mcp-server/dist/index.js',
        '  ANTHROPIC_API_KEY      For claude-* models or judge',
        '  OPENAI_API_KEY         For gpt-* models',
        '',
        'See file header for design rationale and the comparison table.',
      ].join('\n'),
    );
    process.exit(0);
  }

  const retrievalRaw = get('--retrieval', 'mcp-search');
  if (
    retrievalRaw !== 'mcp-search' &&
    retrievalRaw !== 'auto-middleware' &&
    retrievalRaw !== 'agent-loop'
  ) {
    throw new Error(
      `--retrieval must be "mcp-search", "agent-loop", or "auto-middleware"; got: ${retrievalRaw}`,
    );
  }

  const fixtureName = get('--fixture');
  if (!fixtureName) throw new Error('--fixture is required (e.g. data-pipeline)');

  const modelName = get('--model');
  if (!modelName) throw new Error('--model is required (e.g. mock, gpt-4o-mini)');

  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const outputDefault = `/tmp/middleware-replay-${fixtureName}-${modelName}-${ts}`;

  return {
    fixtureName,
    workspaceOverride: get('--workspace'),
    quizOverride: get('--quiz'),
    modelName,
    retrievalMode: retrievalRaw,
    questionsPerSession: parseInt(get('--questions', '0') ?? '0', 10),
    topK: parseInt(get('--top-k', '5') ?? '5', 10),
    seed: parseInt(get('--seed', '1') ?? '1', 10),
    outputDir: get('--output', outputDefault) ?? outputDefault,
    verbose: has('--verbose'),
  };
}

function resolveMcpPath(): string {
  const fromEnv = process.env.CONTINUITY_MCP_PATH;
  if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

  // Convenience auto-detect: walk up from cwd looking for a sibling
  // continuity-ultimate clone.
  const candidates = [
    path.resolve(__dirname, '../../continuity-ultimate/packages/mcp-server/dist/index.js'),
    path.resolve(process.cwd(), '../continuity-ultimate/packages/mcp-server/dist/index.js'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }

  throw new Error(
    [
      'middleware-replay: cannot locate the production MCP server bundle.',
      '',
      'Set the CONTINUITY_MCP_PATH env var to the absolute path of',
      'packages/mcp-server/dist/index.js inside your continuity-ultimate clone:',
      '',
      '  export CONTINUITY_MCP_PATH=/path/to/continuity-ultimate/packages/mcp-server/dist/index.js',
      '',
      'The MCP server bundle is shipped in the commercial workspace, not in',
      'this public benchmarks repo. See file header for setup steps.',
    ].join('\n'),
  );
}

function fixtureWorkspaceRoot(fixtureName: string, override?: string): string {
  if (override) {
    const resolved = path.resolve(override);
    if (!fs.existsSync(path.join(resolved, '.continuity', 'decisions.json'))) {
      throw new Error(
        `--workspace ${override} → ${resolved} is not a Continuity workspace ` +
          `(missing .continuity/decisions.json).`,
      );
    }
    return resolved;
  }
  // The MCP server treats WORKSPACE_ROOT as the project root and looks for
  // `.continuity/decisions.json` under it. Our fixtures are arranged as
  // `fixtures/<name>/.continuity/decisions.json`, so the workspace root is
  // the fixture directory itself.
  const wsRoot = path.resolve(__dirname, '..', 'fixtures', fixtureName);
  if (!fs.existsSync(path.join(wsRoot, '.continuity', 'decisions.json'))) {
    throw new Error(
      `Fixture "${fixtureName}" not found at ${wsRoot} (expected .continuity/decisions.json).`,
    );
  }
  return wsRoot;
}

function formatDecisionsAsContext(decisions: McpDecision[]): string {
  if (decisions.length === 0) return '';
  const lines: string[] = ['## Project decisions (retrieved from Continuity via MCP)'];
  for (const d of decisions) {
    const tagLine = d.tags && d.tags.length > 0 ? ` [${d.tags.join(', ')}]` : '';
    lines.push('');
    lines.push(`### ${d.question}${tagLine}`);
    lines.push(d.answer);
  }
  return lines.join('\n');
}

async function answerQuestion(
  llm: LLMClient,
  question: string,
  context: string,
): Promise<{ answer: string; inputTokens: number; outputTokens: number }> {
  const userMessage = context
    ? `${context}\n\n---\n\nQuestion: ${question}\n\nAnswer concisely with the project's rationale.`
    : `Question: ${question}\n\nAnswer concisely with the project's rationale (you have no project context).`;

  const response = await llm.complete(userMessage, {
    systemPrompt:
      "You are an engineering assistant familiar with this project. Answer the question using the provided project decisions if any are present; otherwise, answer to the best of your ability.",
    maxTokens: 400,
    temperature: 0.2,
  });

  return {
    answer: response.text,
    inputTokens: response.inputTokens,
    outputTokens: response.outputTokens,
  };
}

/**
 * Single-shot retrieval (mode=mcp-search). Calls the production MCP
 * search_decisions tool directly with the prompt as the query, returning
 * the top-K decisions. No agent reasoning involved — this is the
 * "passive RAG" baseline delivered through the production retrieval
 * ranker via real MCP transport.
 */
async function retrieveForQuestionMcpSearch(
  client: McpClient,
  question: string,
  topK: number,
): Promise<{ decisions: McpDecision[]; middlewareFired: boolean }> {
  const result = await client.searchDecisions(question, topK, 'hybrid');
  return { decisions: result.decisions, middlewareFired: false };
}

const BASH_TOOL: ToolDef = {
  name: 'bash',
  description:
    'Run a read-only file-inspection command in the project workspace. ' +
    "Use this to look at the contents of source files relevant to the user's question. " +
    'The Continuity AutoRetrievalMiddleware monitors this tool and surfaces project ' +
    'decisions linked to any file paths in the command. ' +
    'Only `cat <path>` is supported in benchmark mode (no arbitrary execution).',
  inputSchema: {
    type: 'object',
    properties: {
      command: {
        type: 'string',
        description:
          'A shell command of the form `cat <relative-path-to-source-file>`. ' +
          'Example: `cat src/streaming/kafka.ts`. The path must be relative to the project root.',
      },
    },
    required: ['command'],
  },
};

const SEARCH_DECISIONS_TOOL: ToolDef = {
  name: 'search_decisions',
  description:
    'Search the project decision store for entries relevant to a query. ' +
    'Use this when the user asks a "why" question about a project decision, ' +
    'or when you need to ground an answer in past architectural rationale. ' +
    'Returns a list of decisions ranked by hybrid keyword + semantic relevance.',
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'A search query (natural language or keywords)',
      },
      limit: {
        type: 'number',
        description: 'Maximum number of decisions to return (default 5)',
      },
    },
    required: ['query'],
  },
};

/**
 * Convert the MCP search-tool response (a JSON-with-trailing-prose blob
 * inside content[1].text) into a clean string the agent can read in
 * turn 2. We strip the trailing prose footer but keep the JSON results
 * intact — the agent needs the structured data, not the wiki-page hint.
 */
function cleanSearchToolText(rawText: string): string {
  if (!rawText.trim().startsWith('{')) return rawText;
  // Find the closing brace of the leading JSON object (same brace-matching
  // logic as in mcp-client). Anything after is trailing prose.
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < rawText.length; i++) {
    const ch = rawText[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (inString) {
      if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return rawText.slice(0, i + 1);
    }
  }
  return rawText;
}

interface AgentLoopOutcome {
  decisions: McpDecision[];
  middlewareFired: boolean;
  /** The agent's final-turn answer text. */
  finalAnswer: string;
  inputTokens: number;
  outputTokens: number;
  /** True if the agent skipped tooling and answered directly in turn 1. */
  agentAnsweredDirectly: boolean;
}

/**
 * 2-turn agent loop (modes agent-loop and auto-middleware). The agent
 * has one tool advertised; it decides in turn 1 whether to call it, then
 * in turn 2 generates a final answer using the tool result (if any).
 *
 * Returns both the agent's final answer (which is what we score) AND
 * any decisions that were retrieved (for reporting / diagnostics).
 */
async function runAgentLoop(args: {
  agent: ToolCallingAgent;
  client: McpClient;
  question: string;
  groundTruth: string;
  tool: ToolDef;
  topK: number;
}): Promise<AgentLoopOutcome> {
  const systemPrompt =
    "You are an engineering assistant familiar with this project. Use the provided tool to look up relevant project rationale before answering. Once you have the relevant decisions (or have determined none exist), produce a concise answer that reflects the project's documented choices.";

  const turn1 = await args.agent.decideToolCall({
    systemPrompt,
    userMessage: args.question,
    tools: [args.tool],
  });

  // No tool call: agent answered directly in turn 1.
  if (turn1.toolCalls.length === 0) {
    return {
      decisions: [],
      middlewareFired: false,
      finalAnswer: turn1.text,
      inputTokens: turn1.inputTokens,
      outputTokens: turn1.outputTokens,
      agentAnsweredDirectly: true,
    };
  }

  // Dispatch ALL tool calls the agent issued. OpenAI's API rejects a
  // turn-2 messages array unless every tool_call_id from turn 1 has a
  // matching tool result, so we can't just pick the first. Anthropic
  // is similarly strict about block ordering. We've also set
  // parallel_tool_calls: false on OpenAI to encourage single-call
  // turns, but multi-call turns still need to work.
  const dispatches: Array<{ call: typeof turn1.toolCalls[0]; result: Awaited<ReturnType<typeof args.client.dispatchToolCall>> }> = [];
  for (const call of turn1.toolCalls) {
    const r = await args.client.dispatchToolCall(call.name, call.arguments);
    dispatches.push({ call, result: r });
  }

  // Build per-call tool results for turn 2.
  //
  // Format choice (matters for auto-middleware fairness): in production,
  // AutoRetrievalMiddleware delivers `_meta.relevantDecisions` separately
  // from the tool's own output — IDEs typically render the metadata as a
  // distinct context block, not inline with bulk tool output. The text-only
  // tool_result API forces us to concatenate, so we PREPEND decisions
  // (high-signal, agent reads first) and TRUNCATE bulk tool output to
  // 4 KiB (keeps decisions visible vs being buried after a 50 KiB file
  // dump). Without these adjustments, agents pick up the file contents
  // as their primary answer source and ignore the injected decisions —
  // empirically observed to make middleware-fired calls *worse* than
  // non-fired ones.
  const TOOL_OUTPUT_CAP = 4096;
  const toolResults: ToolResult[] = dispatches.map(({ call, result }) => {
    const sections: string[] = [];
    if (result.injectedDecisions.length > 0) {
      const decisionsBlock = result.injectedDecisions
        .map((d) => `- ${d.question}\n  ${d.answer}`)
        .join('\n');
      sections.push(
        `Project decisions linked to this tool call (via Continuity AutoRetrievalMiddleware):\n${decisionsBlock}`,
      );
    }
    let toolBody = cleanSearchToolText(result.contentText);
    if (toolBody.length > TOOL_OUTPUT_CAP) {
      toolBody = toolBody.slice(0, TOOL_OUTPUT_CAP) + '\n\n[…truncated for benchmark; original tool output exceeds 4 KiB]';
    }
    sections.push(toolBody);
    return {
      toolCallId: call.id,
      content: sections.join('\n\n---\n\n'),
      isError: result.isError,
    };
  });

  const turn2 = await args.agent.continueWithToolResults({
    systemPrompt,
    userMessage: args.question,
    tools: [args.tool],
    priorAssistantBlob: turn1.rawAssistantBlob,
    toolResults,
  });

  // Aggregate retrieved decisions for reporting. For search_decisions
  // calls, parse the structured response from each. For bash calls,
  // the relevant decisions are the middleware-injected ones. Concat
  // across all dispatched calls and dedupe by id, then trim to topK.
  const seen = new Set<string>();
  const collected: McpDecision[] = [];
  for (const { call, result } of dispatches) {
    const fromCall = call.name === 'search_decisions'
      ? parseSearchToolDecisions(result.contentText, args.topK)
      : result.injectedDecisions;
    for (const d of fromCall) {
      if (!d.id || seen.has(d.id)) continue;
      seen.add(d.id);
      collected.push(d);
    }
  }
  const decisions = collected.slice(0, args.topK);

  const middlewareFired = dispatches.some((d) => d.result.injectedDecisions.length > 0);
  return {
    decisions,
    middlewareFired,
    finalAnswer: turn2.text,
    inputTokens: turn1.inputTokens + turn2.inputTokens,
    outputTokens: turn1.outputTokens + turn2.outputTokens,
    agentAnsweredDirectly: false,
  };
}

function parseSearchToolDecisions(rawText: string, topK: number): McpDecision[] {
  // The runner's dispatchToolCall joins all text blocks with `\n`. The
  // search_decisions response starts with a warning block, then the
  // JSON-with-trailing-prose block. Scan for the first `{` that opens
  // a balanced JSON object containing `results`.
  const firstBraceIdx = rawText.indexOf('{');
  if (firstBraceIdx < 0) return [];
  const jsonText = cleanSearchToolText(rawText.slice(firstBraceIdx));
  if (!jsonText.trim().startsWith('{')) return [];
  try {
    const parsed = JSON.parse(jsonText) as {
      results?: Array<Partial<McpDecision> & { decisionId?: string }>;
    };
    return (parsed.results ?? []).slice(0, topK).map((d) => ({
      id: d.id ?? d.decisionId ?? '',
      question: d.question ?? '',
      answer: d.answer ?? '',
      tags: d.tags ?? [],
      score: d.score,
    }));
  } catch {
    return [];
  }
}

async function loadQuizFile(
  fixtureName: string,
  override?: string,
): Promise<Array<{ id: string; question: string; groundTruth: string }>> {
  const quizPath = override
    ? path.resolve(override)
    : path.resolve(__dirname, '..', 'prompts', 'quizzes', `${fixtureName}.json`);
  if (!fs.existsSync(quizPath)) {
    throw new Error(`Quiz file not found: ${quizPath}`);
  }
  const raw = JSON.parse(fs.readFileSync(quizPath, 'utf8')) as {
    questions?: Array<{ id: string; question: string; groundTruth?: string; ground_truth?: string }>;
    quiz?: Array<{ id: string; question: string; groundTruth?: string; ground_truth?: string }>;
  } | Array<{ id: string; question: string; groundTruth?: string; ground_truth?: string }>;
  const list = Array.isArray(raw) ? raw : (raw.questions ?? raw.quiz ?? []);
  return list.map((q) => ({
    id: q.id,
    question: q.question,
    groundTruth: q.groundTruth ?? q.ground_truth ?? '',
  }));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const startedAt = Date.now();

  const mcpPath = resolveMcpPath();
  const workspaceRoot = fixtureWorkspaceRoot(args.fixtureName, args.workspaceOverride);
  const quiz = await loadQuizFile(args.fixtureName, args.quizOverride);
  const questions = args.questionsPerSession > 0
    ? quiz.slice(0, args.questionsPerSession)
    : quiz;

  console.log(
    `[middleware-replay] fixture=${args.fixtureName} model=${args.modelName} retrieval=${args.retrievalMode} questions=${questions.length} topK=${args.topK}`,
  );
  console.log(`[middleware-replay] mcp-server: ${mcpPath}`);
  console.log(`[middleware-replay] workspace: ${workspaceRoot}`);

  // mcp-search uses the simple LLMClient (single-shot, no tool use). The
  // two agent-loop modes use ToolCallingAgent.
  const llm =
    args.retrievalMode === 'mcp-search'
      ? createLLMClient(args.modelName as SupportedModel, {
          mock: { responder: (prompt, idx) => `[mock-${idx}] ${prompt.slice(-200)}` },
        })
      : null;
  const agent =
    args.retrievalMode === 'mcp-search' ? null : createAgent(args.modelName);

  const embedder: Embedder = args.modelName === 'mock' ? new MockEmbedder(64) : new EvalEmbedder();
  await embedder.init();
  const client = await McpClient.spawn({
    mcpServerPath: mcpPath,
    workspaceRoot,
    inheritStderr: args.verbose,
    // auto-middleware mode requires the server to register the
    // benchmark-only `bash` tool. CONTINUITY_BENCHMARK_MODE=1 unlocks it.
    env: args.retrievalMode === 'auto-middleware' ? { CONTINUITY_BENCHMARK_MODE: '1' } : {},
  });

  const results: PerQuestionResult[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let middlewareFires = 0;

  // Tool advertised to the agent. agent-loop uses search_decisions;
  // auto-middleware uses the benchmark-mode bash tool (gated server-side
  // on CONTINUITY_BENCHMARK_MODE=1; the server registers a read-only
  // file-cat handler that triggers AutoRetrievalMiddleware).
  const agentTool: ToolDef =
    args.retrievalMode === 'auto-middleware'
      ? BASH_TOOL
      : SEARCH_DECISIONS_TOOL;

  try {
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];

      let decisions: McpDecision[] = [];
      let middlewareFired = false;
      let answer: string;
      let inputTokens: number;
      let outputTokens: number;
      let agentAnsweredDirectly = false;

      if (args.retrievalMode === 'mcp-search') {
        const r = await retrieveForQuestionMcpSearch(client, q.question, args.topK);
        decisions = r.decisions;
        middlewareFired = r.middlewareFired;
        const context = formatDecisionsAsContext(decisions);
        const a = await answerQuestion(llm!, q.question, context);
        answer = a.answer;
        inputTokens = a.inputTokens;
        outputTokens = a.outputTokens;
      } else {
        // agent-loop or auto-middleware: 2-turn agent loop.
        const outcome = await runAgentLoop({
          agent: agent!,
          client,
          question: q.question,
          groundTruth: q.groundTruth,
          tool: agentTool,
          topK: args.topK,
        });
        decisions = outcome.decisions;
        middlewareFired = outcome.middlewareFired;
        answer = outcome.finalAnswer;
        inputTokens = outcome.inputTokens;
        outputTokens = outcome.outputTokens;
        agentAnsweredDirectly = outcome.agentAnsweredDirectly;
      }

      const groundTruthEmbedding = await embedder.embed(q.groundTruth);
      const answerEmbedding = await embedder.embed(answer);
      const cosine = cosineSimilarity(answerEmbedding, groundTruthEmbedding);

      results.push({
        questionId: q.id,
        question: q.question,
        retrievedDecisionIds: decisions.map((d) => d.id),
        middlewareFired,
        agentAnswer: answer,
        cosineVsGroundTruth: cosine,
        inputTokens,
        outputTokens,
      });
      totalInputTokens += inputTokens;
      totalOutputTokens += outputTokens;
      if (middlewareFired) middlewareFires++;

      if (args.verbose) {
        const directNote = agentAnsweredDirectly ? ' [agent-direct]' : '';
        console.log(
          `[middleware-replay] q${i + 1}/${questions.length} ${q.id} → cosine=${cosine.toFixed(3)} retrieved=${decisions.length} middlewareFired=${middlewareFired}${directNote}`,
        );
      }
    }
  } finally {
    await client.close();
  }

  const meanCosine = results.reduce((s, r) => s + r.cosineVsGroundTruth, 0) / Math.max(1, results.length);
  const fractionAbove070 = results.filter((r) => r.cosineVsGroundTruth >= 0.7).length / Math.max(1, results.length);
  const middlewareFireRate = middlewareFires / Math.max(1, results.length);

  const report: RunReport = {
    fixture: args.fixtureName,
    model: args.modelName,
    retrievalMode: args.retrievalMode,
    topK: args.topK,
    seed: args.seed,
    questions: results,
    meanCosine,
    fractionAbove070,
    middlewareFireRate,
    totalInputTokens,
    totalOutputTokens,
    durationMs: Date.now() - startedAt,
  };

  fs.mkdirSync(args.outputDir, { recursive: true });
  const jsonPath = path.join(args.outputDir, 'middleware-replay.json');
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));

  console.log('');
  console.log(`[middleware-replay] complete in ${Math.round(report.durationMs / 1000)}s`);
  console.log(`[middleware-replay]   mean cosine vs ground truth = ${meanCosine.toFixed(4)}`);
  console.log(`[middleware-replay]   fraction ≥ 0.7 = ${(fractionAbove070 * 100).toFixed(1)}%`);
  if (args.retrievalMode === 'auto-middleware') {
    console.log(`[middleware-replay]   middleware fire-rate = ${(middlewareFireRate * 100).toFixed(1)}% (${middlewareFires}/${results.length})`);
  }
  console.log(`[middleware-replay]   tokens: ${totalInputTokens} in / ${totalOutputTokens} out`);
  console.log(`[middleware-replay] report → ${jsonPath}`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
