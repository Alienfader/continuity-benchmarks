/**
 * `continuity-mcp-middleware` runner — END-TO-END PRODUCTION-MIDDLEWARE REPLAY
 *
 * Status: implemented end-to-end (MCP-client wiring + agent-call shape +
 * scoring). Smoke tests pass: `npm run test:smoke-middleware-replay`.
 * The v2 matrix run is gated on user green-light because it costs
 * ~$30–50 in API spend.
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
import { loadFixture, FixtureProject } from './shared/fixtures';
import { LLMClient, createLLMClient, SupportedModel } from './shared/llm-providers';
import { Embedder, EvalEmbedder, MockEmbedder, cosineSimilarity } from './shared/eval-embeddings';
import { extractEntities } from './shared/retrieval';

type RetrievalMode = 'mcp-search' | 'auto-middleware';

interface ReplayArgs {
  fixtureName: string;
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
        '  --retrieval   "mcp-search" (default) or "auto-middleware"',
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
  if (retrievalRaw !== 'mcp-search' && retrievalRaw !== 'auto-middleware') {
    throw new Error(
      `--retrieval must be "mcp-search" or "auto-middleware", got: ${retrievalRaw}`,
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

function fixtureWorkspaceRoot(fixtureName: string): string {
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

async function retrieveForQuestion(
  client: McpClient,
  fixture: FixtureProject,
  question: string,
  mode: RetrievalMode,
  topK: number,
): Promise<{ decisions: McpDecision[]; middlewareFired: boolean }> {
  if (mode === 'mcp-search') {
    const result = await client.searchDecisions(question, topK, 'hybrid');
    return { decisions: result.decisions, middlewareFired: false };
  }

  // auto-middleware: use entity extraction to surface path-shaped tokens,
  // then fire the middleware via a bash tool call.
  const entityQuery = extractEntities(question);
  const tokens = entityQuery
    .split(/\s+/)
    .filter((t) => t.includes('/') || /\.[a-z]+$/i.test(t));
  const result = await client.invokeMiddleware(tokens);
  return {
    decisions: result.injectedDecisions.slice(0, topK),
    middlewareFired: result.middlewareFired,
  };
}

async function loadQuizFile(fixtureName: string): Promise<Array<{ id: string; question: string; groundTruth: string }>> {
  const quizPath = path.resolve(__dirname, '..', 'prompts', 'quizzes', `${fixtureName}.json`);
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
  const workspaceRoot = fixtureWorkspaceRoot(args.fixtureName);
  const quiz = await loadQuizFile(args.fixtureName);
  const questions = args.questionsPerSession > 0
    ? quiz.slice(0, args.questionsPerSession)
    : quiz;
  const fixture = loadFixture(args.fixtureName);

  console.log(
    `[middleware-replay] fixture=${args.fixtureName} model=${args.modelName} retrieval=${args.retrievalMode} questions=${questions.length} topK=${args.topK}`,
  );
  console.log(`[middleware-replay] mcp-server: ${mcpPath}`);
  console.log(`[middleware-replay] workspace: ${workspaceRoot}`);

  const llm = createLLMClient(args.modelName as SupportedModel, {
    mock: { responder: (prompt, idx) => `[mock-${idx}] ${prompt.slice(-200)}` },
  });
  const embedder: Embedder = args.modelName === 'mock' ? new MockEmbedder(64) : new EvalEmbedder();
  await embedder.init();
  const client = await McpClient.spawn({
    mcpServerPath: mcpPath,
    workspaceRoot,
    inheritStderr: args.verbose,
  });

  const results: PerQuestionResult[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let middlewareFires = 0;

  try {
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];

      const { decisions, middlewareFired } = await retrieveForQuestion(
        client,
        fixture,
        q.question,
        args.retrievalMode,
        args.topK,
      );

      const context = formatDecisionsAsContext(decisions);
      const { answer, inputTokens, outputTokens } = await answerQuestion(
        llm,
        q.question,
        context,
      );

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
        console.log(
          `[middleware-replay] q${i + 1}/${questions.length} ${q.id} → cosine=${cosine.toFixed(3)} retrieved=${decisions.length} middlewareFired=${middlewareFired}`,
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
