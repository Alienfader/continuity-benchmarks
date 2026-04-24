/**
 * Benchmark 2/3 — Action Alignment.
 *
 * Does the agent actually USE retrieved context? We generate N "proposed
 * action" prompts per project ("we want to add X — what's your plan?"),
 * run each under the three conditions, and have a Claude Sonnet judge
 * score 1-10 how well the proposed action aligns with the project's
 * decisions.
 *
 * Prompts are synthesised from the fixture's decision topics so the runner
 * works on any project Clio ships.
 *
 * Outputs:
 *   - benchmarks/reports/action-alignment.json
 *   - benchmarks/reports/action-alignment.md
 *
 * Usage:
 *   npx ts-node benchmarks/src/id-rag-parallel/runners/action-alignment.ts \
 *     --fixture paydash-api --model gpt-4o-mini
 */

import * as fs from 'fs';
import * as path from 'path';
import { parseArgs, printHelp } from './shared/cli';
import {
  createLLMClient,
  LLMClient,
  MockLLMClient,
  AnthropicBenchmarkClient,
} from './shared/llm-providers';
import {
  loadFixture,
  ensureReportsDir,
  REPORTS_DIR,
  FixtureProject,
  Decision,
} from './shared/fixtures';
import { BM25Retriever, Condition, renderContext } from './shared/retrieval';

interface ActionResult {
  actionId: string;
  prompt: string;
  condition: Condition;
  proposedAction: string;
  judgeScore: number;
  judgeReasoning: string;
  inputTokens: number;
  outputTokens: number;
  durationMs: number;
}

interface ConditionSummary {
  condition: Condition;
  meanScore: number;
  medianScore: number;
  minScore: number;
  maxScore: number;
  count: number;
}

interface AlignmentReport {
  benchmark: 'action-alignment';
  fixture: string;
  model: string;
  provider: string;
  judgeModel: string;
  actionsPerCondition: number;
  topK: number;
  summaries: ConditionSummary[];
  results: ActionResult[];
  generatedAt: string;
}

// ── Action prompt generation ──────────────────────────────────────────────────

/**
 * Build N action prompts from the fixture. Each prompt reshapes a decision's
 * question into a forward-looking action ("we're about to do X — plan it").
 * This keeps the prompts on-topic without leaking the ground-truth answer.
 */
function buildActionPrompts(fixture: FixtureProject, n: number): string[] {
  const templates: Array<(d: Decision) => string> = [
    (d) => `A new engineer asks: "${verbifyQuestion(d.question)}" — propose a concrete plan with the first three steps. Be specific about tools and rollout order.`,
    (d) => `We're about to kick off work related to: "${trimQuestionPrefix(d.question)}". Write a one-paragraph implementation plan.`,
    (d) => `Product wants a quick answer: "${trimQuestionPrefix(d.question)}". Give them a three-sentence recommendation consistent with our current stack.`,
    (d) => `A junior dev is writing an RFC on "${trimQuestionPrefix(d.question)}". What are the top three constraints you'd flag for them?`,
  ];
  const prompts: string[] = [];
  const decisions = fixture.decisions;
  if (decisions.length === 0) throw new Error(`Fixture ${fixture.name} has no decisions.`);
  for (let i = 0; i < n; i++) {
    const d = decisions[i % decisions.length];
    const tmpl = templates[i % templates.length];
    prompts.push(tmpl(d));
  }
  return prompts;
}

function verbifyQuestion(q: string): string {
  // "Why do we pick X?" → "Should we pick X?"
  return q.replace(/^Why (did|do|does|have|are|is)\s*/i, 'Should ').replace(/\?+$/, '?');
}

function trimQuestionPrefix(q: string): string {
  return q.replace(/^Why\s+/i, '').replace(/\?+$/, '');
}

// ── Judge ─────────────────────────────────────────────────────────────────────

interface Judge {
  score(args: { prompt: string; proposedAction: string; decisions: Decision[] }): Promise<{
    score: number;
    reasoning: string;
  }>;
  getModelName(): string;
}

class ClaudeSonnetJudge implements Judge {
  constructor(private readonly client: LLMClient) {}

  async score(args: { prompt: string; proposedAction: string; decisions: Decision[] }): Promise<{
    score: number;
    reasoning: string;
  }> {
    const decisionBlock = renderContext(args.decisions);
    const judgePrompt = `You are evaluating how well a proposed action aligns with a project's existing architectural decisions.

${decisionBlock}

## The proposed action for this prompt
Prompt: ${args.prompt}

Proposed action:
${args.proposedAction}

## Task
Score 1-10 how well the proposed action aligns with the decisions above. 10 = cites constraints, picks the same tools, is consistent with every decision. 1 = contradicts the decisions or ignores them.

Respond in JSON: {"score": <1-10>, "reasoning": "<one sentence>"}`;

    const resp = await this.client.complete(judgePrompt, {
      systemPrompt: 'You are a careful code reviewer. Always respond with valid JSON.',
      maxTokens: 256,
      temperature: 0,
    });
    return parseJudgeResponse(resp.text);
  }

  getModelName(): string {
    return this.client.getModelName();
  }
}

class MockJudge implements Judge {
  async score(args: { prompt: string; proposedAction: string; decisions: Decision[] }): Promise<{
    score: number;
    reasoning: string;
  }> {
    // Score = fraction of decisions whose distinctive tokens appear in the
    // proposed action, mapped to 1-10. Encodes "did the agent use the
    // retrieved context?" without calling an LLM.
    const tokens = tokenizeForOverlap(args.proposedAction);
    if (args.decisions.length === 0) {
      return { score: 3, reasoning: 'mock judge: no decisions provided as reference' };
    }
    let hits = 0;
    for (const d of args.decisions) {
      const dTokens = tokenizeForOverlap(`${d.question} ${d.answer}`);
      const overlap = [...dTokens].filter((t) => tokens.has(t)).length;
      if (overlap >= 2) hits += 1;
    }
    const frac = hits / args.decisions.length;
    const score = Math.max(1, Math.min(10, Math.round(1 + frac * 9)));
    return {
      score,
      reasoning: `mock judge: ${hits}/${args.decisions.length} decisions echoed in the action (frac=${frac.toFixed(2)})`,
    };
  }
  getModelName(): string {
    return 'mock-judge';
  }
}

function parseJudgeResponse(text: string): { score: number; reasoning: string } {
  // Be forgiving: extract the first {...} block
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    return { score: 0, reasoning: `could not parse judge response: ${text.slice(0, 200)}` };
  }
  try {
    const parsed = JSON.parse(match[0]) as { score?: unknown; reasoning?: unknown };
    const score = Number(parsed.score);
    const reasoning = typeof parsed.reasoning === 'string' ? parsed.reasoning : '';
    if (!Number.isFinite(score) || score < 0 || score > 10) {
      return { score: 0, reasoning: `judge score out of range: ${parsed.score}` };
    }
    return { score, reasoning };
  } catch (e) {
    return { score: 0, reasoning: `judge response JSON parse failed: ${String(e)}` };
  }
}

const STOPWORDS_OVERLAP = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'by', 'with',
  'is', 'are', 'was', 'we', 'our', 'they', 'it', 'as', 'at', 'this', 'that',
  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
]);

function tokenizeForOverlap(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s\-_]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 3 && !STOPWORDS_OVERLAP.has(t)),
  );
}

// ── Runner ────────────────────────────────────────────────────────────────────

async function askAgent(
  client: LLMClient,
  prompt: string,
  contextBlock: string,
): Promise<{ text: string; inputTokens: number; outputTokens: number; durationMs: number }> {
  const sys =
    'You are a senior engineer on the project. Propose concrete, specific actions. Reference known project decisions where relevant.';
  const full = contextBlock ? `${contextBlock}\n\n---\n\n${prompt}` : prompt;
  const r = await client.complete(full, { systemPrompt: sys, maxTokens: 400, temperature: 0.3 });
  return { text: r.text, inputTokens: r.inputTokens, outputTokens: r.outputTokens, durationMs: r.durationMs };
}

function buildMockAgentResponder(retriever: BM25Retriever): (prompt: string) => string {
  return (prompt: string) => {
    const parts = prompt.split('---');
    const hasContext = parts.length >= 2;
    if (hasContext) {
      const decisions = retriever.retrieve(parts[parts.length - 1], 3);
      const keywords = decisions
        .map((d) => d.question.replace(/^Why\s+/i, '').replace(/\?+$/, ''))
        .join(', ');
      return `Plan: extend the existing approach (${keywords}) and add incremental tests. Rollout: behind flag, measure latency, expand gradually.`;
    }
    return `Plan: start fresh, try multiple options, pick one based on which feels fastest. No particular constraints.`;
  };
}

function summarize(results: ActionResult[], condition: Condition): ConditionSummary {
  const scores = results.filter((r) => r.condition === condition).map((r) => r.judgeScore);
  if (scores.length === 0) {
    return { condition, meanScore: 0, medianScore: 0, minScore: 0, maxScore: 0, count: 0 };
  }
  const sorted = [...scores].sort((a, b) => a - b);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const median =
    sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[(sorted.length - 1) / 2];
  return {
    condition,
    meanScore: mean,
    medianScore: median,
    minScore: sorted[0],
    maxScore: sorted[sorted.length - 1],
    count: scores.length,
  };
}

function writeReport(report: AlignmentReport, outputBase: string): { json: string; md: string } {
  const jsonPath = `${outputBase}.json`;
  const mdPath = `${outputBase}.md`;
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf-8');
  fs.writeFileSync(mdPath, renderMarkdown(report), 'utf-8');
  return { json: jsonPath, md: mdPath };
}

function renderMarkdown(r: AlignmentReport): string {
  const lines: string[] = [];
  lines.push(`# Action Alignment — ${r.fixture} / ${r.model}`);
  lines.push('');
  lines.push(`- Generated: ${r.generatedAt}`);
  lines.push(`- Agent model: ${r.model} (${r.provider})`);
  lines.push(`- Judge model: ${r.judgeModel}`);
  lines.push(`- Actions per condition: ${r.actionsPerCondition}`);
  lines.push(`- Retrieval top-K: ${r.topK}`);
  lines.push('');
  lines.push('## Summary (judge score 1-10, higher is better)');
  lines.push('');
  lines.push('| Condition | Mean | Median | Min | Max | N |');
  lines.push('|---|---|---|---|---|---|');
  for (const s of r.summaries) {
    lines.push(
      `| ${s.condition} | ${s.meanScore.toFixed(2)} | ${s.medianScore.toFixed(2)} | ${s.minScore} | ${s.maxScore} | ${s.count} |`,
    );
  }
  lines.push('');
  lines.push('## Interpretation');
  lines.push('');
  lines.push(
    '- A large gap between `baseline` and `continuity` indicates the agent uses retrieved context (and the judge recognises alignment).',
  );
  lines.push(
    '- `continuity-in-loop` vs `continuity` tells you whether per-question re-retrieval is worth the extra tool call.',
  );
  return lines.join('\n') + '\n';
}

// ── Entrypoint ────────────────────────────────────────────────────────────────

export async function main(argv = process.argv.slice(2)): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp('action-alignment', [
      '  --actions <n>           Prompts per condition (default: 30)',
      '  --topK <n>              Retrieval top-K (default: 5)',
    ]);
    return;
  }

  const args = parseArgs(argv);
  const actions = args.actions ?? 30;
  const topK = args.topK ?? 5;

  const fixture = loadFixture(args.fixture);
  const retriever = new BM25Retriever(fixture.decisions);
  const actionPrompts = buildActionPrompts(fixture, actions);

  const useMock = args.mock || args.model === 'mock';
  const client: LLMClient = useMock
    ? new MockLLMClient({ latencyMs: 0, responder: buildMockAgentResponder(retriever) })
    : createLLMClient(args.model);

  // Judge is always Claude Sonnet unless we're in mock mode.
  const judge: Judge = useMock
    ? new MockJudge()
    : new ClaudeSonnetJudge(new AnthropicBenchmarkClient('claude-sonnet-4-6'));

  console.log(
    `[action-alignment] fixture=${fixture.name} decisions=${fixture.decisions.length} agent=${client.getModelName()} judge=${judge.getModelName()} actions=${actions}`,
  );

  const results: ActionResult[] = [];
  for (let i = 0; i < actionPrompts.length; i++) {
    const prompt = actionPrompts[i];
    for (const condition of args.conditions) {
      let contextBlock = '';
      let retrieved: Decision[] = [];
      if (condition === 'continuity') {
        retrieved = retriever.retrieve(prompt, topK);
        contextBlock = renderContext(retrieved);
      } else if (condition === 'continuity-in-loop') {
        // Simulate re-retrieving right before the action, once per turn.
        retrieved = retriever.retrieve(prompt, topK);
        contextBlock = renderContext(retrieved);
      }

      const agentResp = await askAgent(client, prompt, contextBlock);
      const judged = await judge.score({
        prompt,
        proposedAction: agentResp.text,
        decisions: retrieved.length > 0 ? retrieved : retriever.retrieve(prompt, topK),
      });

      results.push({
        actionId: `A${i + 1}`,
        prompt,
        condition,
        proposedAction: agentResp.text,
        judgeScore: judged.score,
        judgeReasoning: judged.reasoning,
        inputTokens: agentResp.inputTokens,
        outputTokens: agentResp.outputTokens,
        durationMs: agentResp.durationMs,
      });

      if (args.verbose) {
        console.log(
          `[action-alignment A${i + 1} ${condition}] score=${judged.score} (${judged.reasoning.slice(0, 80)})`,
        );
      }
    }
  }

  const summaries = args.conditions.map((c) => summarize(results, c));
  const report: AlignmentReport = {
    benchmark: 'action-alignment',
    fixture: fixture.name,
    model: client.getModelName(),
    provider: client.getProviderName(),
    judgeModel: judge.getModelName(),
    actionsPerCondition: actions,
    topK,
    summaries,
    results,
    generatedAt: new Date().toISOString(),
  };

  ensureReportsDir();
  const base = args.output ?? path.join(REPORTS_DIR, 'action-alignment');
  const paths = writeReport(report, base);
  console.log(`[action-alignment] wrote ${paths.json}`);
  console.log(`[action-alignment] wrote ${paths.md}`);
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
