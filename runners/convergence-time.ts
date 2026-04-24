/**
 * Benchmark 3/3 — Convergence Time.
 *
 * An efficiency proxy modelled on the ID-RAG paper's "convergence" metric.
 * We simulate a fixed 5-step refactor task per project. Each step is a
 * discrete prompt — e.g., "list the files that need to change", "draft the
 * new module layout", "enumerate test coverage changes", etc. For each
 * condition we measure:
 *
 *   - number of steps the agent took to produce a converged plan
 *   - total input + output tokens
 *   - total wall-clock seconds
 *
 * Agent is Claude Sonnet (via the Anthropic SDK) unless --model overrides.
 * A "tool call" in this benchmark = one LLM round-trip (we're running the
 * steps serially, which is the worst-case shape for convergence).
 *
 * Conditions:
 *   - baseline                 : no .continuity context
 *   - continuity               : top-K retrieved once, prepended to every step
 *   - continuity-in-loop       : retrieved FRESH before each of the 5 steps
 *
 * Outputs:
 *   - benchmarks/reports/convergence-time.json
 *   - benchmarks/reports/convergence-time.md
 */

import * as fs from 'fs';
import * as path from 'path';
import { parseArgs, printHelp } from './shared/cli';
import {
  createLLMClient,
  LLMClient,
  MockLLMClient,
} from './shared/llm-providers';
import {
  loadFixture,
  ensureReportsDir,
  REPORTS_DIR,
  FixtureProject,
  Decision,
} from './shared/fixtures';
import { BM25Retriever, Condition, renderContext } from './shared/retrieval';

interface StepRecord {
  stepIdx: number;
  promptSummary: string;
  response: string;
  inputTokens: number;
  outputTokens: number;
  durationMs: number;
  converged: boolean;
}

interface ConditionRun {
  condition: Condition;
  steps: StepRecord[];
  totalSteps: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalDurationMs: number;
  /** steps-until-converged (if any). For agents that converge early we stop. */
  stepsToConverge: number;
  totalWallClockMinutes: number;
}

interface ConvergenceReport {
  benchmark: 'convergence-time';
  fixture: string;
  model: string;
  provider: string;
  taskName: string;
  runs: ConditionRun[];
  generatedAt: string;
}

// ── Task definition ───────────────────────────────────────────────────────────

/** The 5 canonical steps of the refactor task, written as fixture-agnostic prompts. */
const REFACTOR_STEPS: Array<{ name: string; prompt: (fixtureName: string) => string }> = [
  {
    name: 'survey',
    prompt: (f) =>
      `We're planning a refactor in the ${f} codebase: extract the authorization middleware into a reusable module. Survey the project's architecture and list the three files most likely to be affected. Output JSON: { "files": [...], "notes": "..." }.`,
  },
  {
    name: 'design',
    prompt: () =>
      `Based on the survey, propose the new module\'s public API — exported functions, their signatures, and the injection boundary. Output JSON: { "api": [...] }.`,
  },
  {
    name: 'migration',
    prompt: () =>
      `Sketch the step-by-step migration of the first affected file: what to extract, what to leave, how to keep tests passing. Output JSON: { "steps": [...] }.`,
  },
  {
    name: 'tests',
    prompt: () =>
      `List the test cases that must be added or changed to cover the new module. Keep the list minimal. Output JSON: { "tests": [...] }.`,
  },
  {
    name: 'rollout',
    prompt: () =>
      `Finalize the rollout plan: feature-flag strategy, canary rollout steps, and rollback trigger. Emit CONVERGED: yes on a final line if this plan is ready to execute. Output JSON: { "plan": "..." }.`,
  },
];

// ── Agent invocation ──────────────────────────────────────────────────────────

async function runOneStep(
  client: LLMClient,
  stepPrompt: string,
  contextBlock: string,
): Promise<{ text: string; inputTokens: number; outputTokens: number; durationMs: number }> {
  const sys =
    'You are a senior engineer running a refactor task. Produce concise, actionable output. When a step\'s plan is ready, emit CONVERGED: yes.';
  const full = contextBlock ? `${contextBlock}\n\n---\n\n${stepPrompt}` : stepPrompt;
  const r = await client.complete(full, { systemPrompt: sys, maxTokens: 512, temperature: 0.2 });
  return { text: r.text, inputTokens: r.inputTokens, outputTokens: r.outputTokens, durationMs: r.durationMs };
}

function detectConverged(text: string, stepIdx: number): boolean {
  if (/CONVERGED:\s*yes/i.test(text)) return true;
  // The last step implicitly converges if the agent produced a rollout plan
  return stepIdx === REFACTOR_STEPS.length - 1;
}

function buildMockResponder(fixture: FixtureProject, retriever: BM25Retriever): (prompt: string) => string {
  return (prompt: string) => {
    const hasContext = /## Project decisions/.test(prompt);
    // Count tokens in the context block, if any. An agent with context needs
    // fewer clarification round-trips → converges faster.
    const stepHint = REFACTOR_STEPS.find((s) =>
      prompt.includes(s.prompt(fixture.name).slice(0, 40)),
    )?.name ?? 'generic';
    const top = retriever.retrieve(prompt, 3);
    const tokensFromDecisions = top.map((d) => d.question.replace(/^Why\s+/i, '').slice(0, 40)).join('; ');
    if (hasContext) {
      if (stepHint === 'rollout') return `Plan ready. ${tokensFromDecisions}. CONVERGED: yes`;
      return `Based on project decisions: ${tokensFromDecisions}. Next step ready.`;
    }
    // Without context, pretend to meander: do not converge until step 5.
    if (stepHint === 'rollout') return `Plan drafted. Feature flag TBD. CONVERGED: yes`;
    return `Exploring options — need more info about tools/conventions before committing to a plan.`;
  };
}

// ── Runner ────────────────────────────────────────────────────────────────────

async function runCondition(args: {
  condition: Condition;
  client: LLMClient;
  fixture: FixtureProject;
  retriever: BM25Retriever;
  topK: number;
  verbose?: boolean;
  maxSteps: number;
}): Promise<ConditionRun> {
  const { condition, client, fixture, retriever, topK, verbose, maxSteps } = args;
  const t0 = Date.now();
  const steps: StepRecord[] = [];
  let totalIn = 0;
  let totalOut = 0;
  let totalDuration = 0;
  let stepsToConverge = -1;

  // Upfront retrieval once, for `continuity` condition.
  const upfrontSeed = fixture.decisions.map((d) => d.question).join(' ');
  const upfrontDecisions = condition !== 'baseline' ? retriever.retrieve(upfrontSeed, topK) : [];
  const upfrontContext = condition === 'continuity' ? renderContext(upfrontDecisions) : '';

  const totalSteps = Math.min(maxSteps, REFACTOR_STEPS.length);

  for (let i = 0; i < totalSteps; i++) {
    const step = REFACTOR_STEPS[i];
    const stepPrompt = step.prompt(fixture.name);

    let contextBlock = upfrontContext;
    if (condition === 'continuity-in-loop') {
      const top = retriever.retrieve(stepPrompt, topK);
      contextBlock = renderContext(top);
    }

    const resp = await runOneStep(client, stepPrompt, contextBlock);
    const converged = detectConverged(resp.text, i);
    const record: StepRecord = {
      stepIdx: i,
      promptSummary: step.name,
      response: resp.text,
      inputTokens: resp.inputTokens,
      outputTokens: resp.outputTokens,
      durationMs: resp.durationMs,
      converged,
    };
    steps.push(record);
    totalIn += resp.inputTokens;
    totalOut += resp.outputTokens;
    totalDuration += resp.durationMs;
    if (verbose) {
      console.log(
        `[convergence-time ${condition} step=${step.name}] tokens=${resp.inputTokens}+${resp.outputTokens} dur=${resp.durationMs}ms converged=${converged}`,
      );
    }
    if (converged && stepsToConverge === -1) stepsToConverge = i + 1;
  }

  if (stepsToConverge === -1) stepsToConverge = totalSteps; // never converged → max

  const totalWall = (Date.now() - t0) / 60000;
  return {
    condition,
    steps,
    totalSteps,
    totalInputTokens: totalIn,
    totalOutputTokens: totalOut,
    totalDurationMs: totalDuration,
    stepsToConverge,
    totalWallClockMinutes: totalWall,
  };
}

// ── Report ────────────────────────────────────────────────────────────────────

function writeReport(r: ConvergenceReport, outputBase: string): { json: string; md: string } {
  const jsonPath = `${outputBase}.json`;
  const mdPath = `${outputBase}.md`;
  fs.writeFileSync(jsonPath, JSON.stringify(r, null, 2), 'utf-8');
  fs.writeFileSync(mdPath, renderMarkdown(r), 'utf-8');
  return { json: jsonPath, md: mdPath };
}

function renderMarkdown(r: ConvergenceReport): string {
  const lines: string[] = [];
  lines.push(`# Convergence Time — ${r.fixture} / ${r.model}`);
  lines.push('');
  lines.push(`- Generated: ${r.generatedAt}`);
  lines.push(`- Agent: ${r.model} (${r.provider})`);
  lines.push(`- Task: ${r.taskName}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Condition | Steps to converge | Total input tok | Total output tok | Wall-clock min |');
  lines.push('|---|---|---|---|---|');
  for (const run of r.runs) {
    lines.push(
      `| ${run.condition} | ${run.stepsToConverge}/${run.totalSteps} | ${run.totalInputTokens} | ${run.totalOutputTokens} | ${run.totalWallClockMinutes.toFixed(2)} |`,
    );
  }
  lines.push('');
  lines.push('## Interpretation');
  lines.push('');
  lines.push('- Lower `steps to converge` = agent reaches a rollout-ready plan sooner.');
  lines.push('- Continuity conditions that cut token consumption prove retrieval compresses context.');
  lines.push(
    '- If `continuity-in-loop` converges in FEWER steps than `continuity` despite higher per-step cost, the mid-task retrieval is net-positive.',
  );
  return lines.join('\n') + '\n';
}

// ── Entrypoint ────────────────────────────────────────────────────────────────

export async function main(argv = process.argv.slice(2)): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp('convergence-time', [
      '  --steps <n>             Max refactor steps (default: 5, clamped to the fixed plan)',
      '  --topK <n>              Retrieval top-K (default: 5)',
    ]);
    return;
  }

  const args = parseArgs(argv);
  const maxSteps = args.steps ?? REFACTOR_STEPS.length;
  const topK = args.topK ?? 5;

  const fixture = loadFixture(args.fixture);
  const retriever = new BM25Retriever(fixture.decisions);

  const useMock = args.mock || args.model === 'mock';
  // Default to claude-sonnet-4-6 (real mode) per the spec.
  const resolvedModel = useMock
    ? 'mock'
    : args.model === 'mock'
      ? 'claude-sonnet-4-6'
      : args.model;
  const client: LLMClient = useMock
    ? new MockLLMClient({ latencyMs: 0, responder: buildMockResponder(fixture, retriever) })
    : createLLMClient(resolvedModel as Parameters<typeof createLLMClient>[0]);

  console.log(
    `[convergence-time] fixture=${fixture.name} agent=${client.getModelName()} conditions=${args.conditions.join(',')}`,
  );

  const runs: ConditionRun[] = [];
  for (const condition of args.conditions) {
    console.log(`[convergence-time] condition=${condition}`);
    const run = await runCondition({
      condition,
      client,
      fixture,
      retriever,
      topK,
      verbose: args.verbose,
      maxSteps,
    });
    runs.push(run);
  }

  const report: ConvergenceReport = {
    benchmark: 'convergence-time',
    fixture: fixture.name,
    model: client.getModelName(),
    provider: client.getProviderName(),
    taskName: 'extract-authorization-middleware',
    runs,
    generatedAt: new Date().toISOString(),
  };

  ensureReportsDir();
  const base = args.output ?? path.join(REPORTS_DIR, 'convergence-time');
  const paths = writeReport(report, base);
  console.log(`[convergence-time] wrote ${paths.json}`);
  console.log(`[convergence-time] wrote ${paths.md}`);
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}

// Re-exported helper so non-CLI callers can discover how many steps exist.
export const TOTAL_REFACTOR_STEPS = REFACTOR_STEPS.length;
