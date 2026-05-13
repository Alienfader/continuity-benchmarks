/**
 * Benchmark 1/3 — Recall over Time.
 *
 * Mirrors the ID-RAG paper's "identity drift" protocol. For each
 * project × model × condition we run N sessions back-to-back. Between
 * sessions we inject ~5k tokens of off-topic noise (simulates unrelated
 * agent work "elsewhere in the world"). At every session boundary we ask
 * all Q quiz questions and score each answer against ground truth with
 * cosine similarity over all-mpnet-base-v2 embeddings.
 *
 * Conditions (this runner uses the 4-condition expanded matrix; see
 * shared/retrieval.ts for the canonical definitions):
 *
 *   - baseline                       no project context
 *   - continuity-blanket             retrieval keyed on the concatenation of
 *                                    all 20 question stems (project-level
 *                                    seed); same context every session
 *   - continuity-perq-frontloaded    M2 ablation: per-question retrieval
 *                                    computed ONCE at session 1, the same
 *                                    20 question-specific blobs re-injected
 *                                    every session boundary
 *   - continuity-in-loop             per-question retrieval done FRESH at
 *                                    every session boundary (re-fires under
 *                                    noise drift)
 *   - continuity                     legacy alias for continuity-blanket;
 *                                    accepted on input for backwards compat
 *                                    with the v6.3 runs
 *
 * Pairwise contrasts a reviewer can read from this run:
 *   - baseline → blanket           : effect of any retrieval
 *   - blanket → perq-frontloaded   : effect of better keying (timing held)
 *   - perq-frontloaded → in-loop   : effect of fresh re-retrieval (timing only)
 *
 * Outputs:
 *   - benchmarks/reports/recall-over-time.json  (machine-parseable)
 *   - benchmarks/reports/recall-over-time.md    (human summary)
 *
 * Usage:
 *   npx ts-node benchmarks/src/id-rag-parallel/runners/recall-over-time.ts \
 *     --fixture paydash-api --model gpt-4o-mini
 *
 * Default model is `mock` so the runner is safe to smoke-test without a key.
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
  EvalEmbedder,
  Embedder,
  MockEmbedder,
  scoreAnswer,
  summarizeScores,
  RecallSummary,
} from './shared/eval-embeddings';
import { generateNoise } from './shared/noise-generator';
import {
  loadFixture,
  loadQuiz,
  ensureReportsDir,
  REPORTS_DIR,
  Quiz,
  QuizQuestion,
} from './shared/fixtures';
import { BM25Retriever, Condition, Retriever, renderContext } from './shared/retrieval';
import { loadSystem } from './shared/system-adapter';

interface PerQuestionRecord {
  sessionIdx: number;
  questionId: string;
  question: string;
  predicted: string;
  groundTruth: string;
  score: number;
  inputTokens: number;
  outputTokens: number;
  durationMs: number;
}

interface ConditionReport {
  condition: Condition;
  sessionSummaries: Array<{ sessionIdx: number; summary: RecallSummary }>;
  perQuestion: PerQuestionRecord[];
  /** Average score across all sessions (the single "recall" number). */
  meanAcrossSessions: number;
  /** Slope (cosine sim drop per session). Negative = drift. */
  driftSlope: number;
}

interface RecallReport {
  benchmark: 'recall-over-time';
  fixture: string;
  model: string;
  provider: string;
  seed: number;
  sessions: number;
  topK: number;
  quizSize: number;
  conditions: ConditionReport[];
  generatedAt: string;
  /** Custom retrieval system name when --system=<name> was used. */
  system?: string;
}

// ── Mock responder used when --model=mock ─────────────────────────────────────

/**
 * Mock responder that tries to "remember" context: if the context block
 * mentions the question's ground-truth keywords, return something close to
 * the ground truth. Otherwise return a generic "I don't know" answer that
 * scores low. This makes the baseline vs continuity difference visible
 * during a dry-run without needing a real model.
 */
function buildMockResponder(
  quiz: Quiz,
  noisePerSession: string[],
): (prompt: string, callIndex: number) => string {
  return (prompt: string, callIndex: number) => {
    // Pick which question we're answering from the prompt
    for (const q of quiz.questions) {
      if (prompt.includes(q.question)) {
        // If the ground-truth is visible in the prompt (retrieved context),
        // respond with a lightly paraphrased version of it.
        const gt = q.groundTruth;
        const gtShort = gt.slice(0, 120);
        if (prompt.toLowerCase().includes(gtShort.toLowerCase().slice(0, 40))) {
          return `Based on the project decisions: ${gt}`;
        }
        // No context — emit a drifted placeholder dependent on callIndex so
        // different sessions yield slightly different answers.
        return `I am not sure, but I think the answer is generic response #${callIndex}.`;
      }
    }
    return `fallback-${callIndex}`;
  };
}

// ── Core runner ───────────────────────────────────────────────────────────────

async function askOnce(
  client: LLMClient,
  question: QuizQuestion,
  contextBlock: string,
): Promise<{ text: string; inputTokens: number; outputTokens: number; durationMs: number }> {
  const systemPrompt =
    'You are an engineer on the project. Answer the following question based on everything you know about the project. Be specific. If you do not know, say so.';
  const prompt = contextBlock
    ? `${contextBlock}\n\n---\n\n${question.question}`
    : question.question;
  const r = await client.complete(prompt, { systemPrompt, maxTokens: 512, temperature: 0.2 });
  return {
    text: r.text,
    inputTokens: r.inputTokens,
    outputTokens: r.outputTokens,
    durationMs: r.durationMs,
  };
}

async function runCondition(args: {
  condition: Condition;
  client: LLMClient;
  embedder: Embedder;
  quiz: Quiz;
  retriever: Retriever;
  sessions: number;
  topK: number;
  seed: number;
  verbose?: boolean;
}): Promise<ConditionReport> {
  const { condition, client, embedder, quiz, retriever, sessions, topK, seed, verbose } = args;

  const perQuestion: PerQuestionRecord[] = [];
  const sessionSummaries: Array<{ sessionIdx: number; summary: RecallSummary }> = [];

  // Set up retrieval state per condition. See shared/retrieval.ts for the
  // canonical definitions of each condition.
  //
  // - baseline                    : no retrieval
  // - continuity-blanket          : single retrieval on concatenated quiz
  //                                 stems; same blob every session/question
  // - continuity-perq-frontloaded : per-question retrievals computed ONCE at
  //                                 the start; the same 20 blobs are
  //                                 re-injected unchanged every session
  // - continuity-in-loop          : per-question retrieval computed FRESH at
  //                                 every session boundary
  // - continuity                  : legacy alias → continuity-blanket
  const isBlanket = condition === 'continuity-blanket' || condition === 'continuity';
  const isPerQFrontloaded = condition === 'continuity-perq-frontloaded';
  const isInLoop = condition === 'continuity-in-loop';

  const upfrontSeed = quiz.questions.map((q) => q.question).join(' ');
  const blanketContext = isBlanket
    ? renderContext(retriever.retrieve(upfrontSeed, topK))
    : '';

  // Per-question retrievals computed once at session 1 and frozen for the
  // whole run — the M2 ablation comparand to in-loop's fresh-retrieval.
  const frontloadedPerQ: Map<string, string> = new Map();
  if (isPerQFrontloaded) {
    for (const q of quiz.questions) {
      frontloadedPerQ.set(q.id, renderContext(retriever.retrieve(q.question, topK)));
    }
  }

  for (let s = 0; s < sessions; s++) {
    if (s > 0) {
      // Noise between sessions. We discard the text — in this simulation we
      // only need its *existence* to model "unrelated turns" between quiz
      // boundaries. If we were exercising a real agent loop with a persistent
      // context window we'd pass it through; here we simply count it as the
      // reason the LLM has "forgotten" the project (mock client) or, with a
      // real model, the reason we re-retrieve.
      generateNoise({ targetTokens: 5000, seed: seed + s * 17 });
    }

    const scores: number[] = [];
    for (const q of quiz.questions) {
      let contextBlock = blanketContext;
      if (isPerQFrontloaded) {
        contextBlock = frontloadedPerQ.get(q.id) ?? '';
      } else if (isInLoop) {
        const top = retriever.retrieve(q.question, topK);
        contextBlock = renderContext(top);
      }
      const resp = await askOnce(client, q, contextBlock);
      const score = await scoreAnswer(resp.text, q.groundTruth, embedder);
      scores.push(score);
      perQuestion.push({
        sessionIdx: s,
        questionId: q.id,
        question: q.question,
        predicted: resp.text,
        groundTruth: q.groundTruth,
        score,
        inputTokens: resp.inputTokens,
        outputTokens: resp.outputTokens,
        durationMs: resp.durationMs,
      });
      if (verbose) {
        console.log(
          `[${condition} s${s} ${q.id}] score=${score.toFixed(3)} tokens=${resp.inputTokens}+${resp.outputTokens}`,
        );
      }
    }
    sessionSummaries.push({ sessionIdx: s, summary: summarizeScores(scores) });
  }

  const meanAcrossSessions =
    sessionSummaries.reduce((sum, s) => sum + s.summary.mean, 0) / Math.max(1, sessionSummaries.length);
  const driftSlope = linearSlope(sessionSummaries.map((s) => s.summary.mean));

  return {
    condition,
    sessionSummaries,
    perQuestion,
    meanAcrossSessions,
    driftSlope,
  };
}

function linearSlope(ys: number[]): number {
  const n = ys.length;
  if (n < 2) return 0;
  const xs = ys.map((_, i) => i);
  const meanX = (n - 1) / 2;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - meanX) * (ys[i] - meanY);
    den += (xs[i] - meanX) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

// ── Report writers ────────────────────────────────────────────────────────────

function writeReport(report: RecallReport, outputBase: string): { json: string; md: string } {
  const jsonPath = `${outputBase}.json`;
  const mdPath = `${outputBase}.md`;
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf-8');
  fs.writeFileSync(mdPath, renderMarkdown(report), 'utf-8');
  return { json: jsonPath, md: mdPath };
}

function renderMarkdown(r: RecallReport): string {
  const lines: string[] = [];
  lines.push(`# Recall-over-Time — ${r.fixture} / ${r.model}`);
  lines.push('');
  lines.push(`- Generated: ${r.generatedAt}`);
  lines.push(`- Provider: ${r.provider}`);
  lines.push(`- Sessions: ${r.sessions}`);
  lines.push(`- Quiz questions: ${r.quizSize}`);
  lines.push(`- Retrieval top-K: ${r.topK}`);
  lines.push(`- Seed: ${r.seed}`);
  lines.push('');
  lines.push('## Summary (mean cosine similarity vs ground truth)');
  lines.push('');
  lines.push('| Condition | Mean | Drift slope / session | Frac ≥ 0.7 (last session) |');
  lines.push('|---|---|---|---|');
  for (const c of r.conditions) {
    const last = c.sessionSummaries[c.sessionSummaries.length - 1];
    lines.push(
      `| ${c.condition} | ${c.meanAcrossSessions.toFixed(3)} | ${c.driftSlope.toFixed(4)} | ${last ? last.summary.fractionAbove070.toFixed(2) : '—'} |`,
    );
  }
  lines.push('');
  lines.push('## Per-session means');
  lines.push('');
  const header = ['Session', ...r.conditions.map((c) => c.condition)];
  lines.push('| ' + header.join(' | ') + ' |');
  lines.push('|' + header.map(() => '---').join('|') + '|');
  for (let s = 0; s < r.sessions; s++) {
    const row = [String(s)];
    for (const c of r.conditions) {
      const sess = c.sessionSummaries.find((x) => x.sessionIdx === s);
      row.push(sess ? sess.summary.mean.toFixed(3) : '—');
    }
    lines.push('| ' + row.join(' | ') + ' |');
  }
  lines.push('');
  lines.push('## Interpretation');
  lines.push('');
  lines.push(
    '- A flat slope near 0 with a high mean indicates the condition retains project context across sessions.',
  );
  lines.push(
    '- A steep negative slope under `baseline` with a shallower slope under `continuity` is the ID-RAG-equivalent drift reduction signal.',
  );
  lines.push(
    '- `continuity-in-loop` should dominate when sessions are long or the quiz mixes direct-recall with supersedes-aware questions.',
  );
  return lines.join('\n') + '\n';
}

// ── Entrypoint ────────────────────────────────────────────────────────────────

export async function main(argv = process.argv.slice(2)): Promise<void> {
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp('recall-over-time', [
      '  --sessions <n>          Number of sequential sessions (default: 7)',
      '  --topK <n>              Decisions retrieved per session (default: 5)',
    ]);
    return;
  }

  const args = parseArgs(argv);
  const sessions = args.sessions ?? 7;
  const topK = args.topK ?? 5;

  const fixture = loadFixture(args.fixture);
  const quiz = loadQuiz(args.fixture, fixture);
  let retriever: Retriever = new BM25Retriever(fixture.decisions);
  let systemLabel: string | undefined;
  if (args.system) {
    const projectRoot = path.resolve(__dirname, '..');
    const adapter = await loadSystem(args.system, projectRoot);
    retriever = await adapter.init(fixture.decisions);
    systemLabel = adapter.name;
    console.log(
      `[recall-over-time] system=${adapter.name}${adapter.description ? ` (${adapter.description})` : ''}`,
    );
  }

  const useMock = args.mock || args.model === 'mock';
  const mockResponder = useMock
    ? buildMockResponder(quiz, Array(sessions).fill(''))
    : undefined;
  const client: LLMClient = useMock
    ? new MockLLMClient({ latencyMs: 0, responder: mockResponder })
    : createLLMClient(args.model);

  const embedder: Embedder = useMock ? new MockEmbedder(64) : new EvalEmbedder();

  console.log(
    `[recall-over-time] fixture=${fixture.name} decisions=${fixture.decisions.length} model=${client.getModelName()} conditions=${args.conditions.join(',')}`,
  );

  const conditionReports: ConditionReport[] = [];
  for (const condition of args.conditions) {
    console.log(`[recall-over-time] condition=${condition}`);
    const report = await runCondition({
      condition,
      client,
      embedder,
      quiz,
      retriever,
      sessions,
      topK,
      seed: args.seed,
      verbose: args.verbose,
    });
    conditionReports.push(report);
  }

  const report: RecallReport = {
    benchmark: 'recall-over-time',
    fixture: fixture.name,
    model: client.getModelName(),
    provider: client.getProviderName(),
    seed: args.seed,
    sessions,
    topK,
    quizSize: quiz.questions.length,
    conditions: conditionReports,
    generatedAt: new Date().toISOString(),
    ...(systemLabel ? { system: systemLabel } : {}),
  };

  ensureReportsDir();
  const base = args.output ?? path.join(REPORTS_DIR, 'recall-over-time');
  const paths = writeReport(report, base);
  console.log(`[recall-over-time] wrote ${paths.json}`);
  console.log(`[recall-over-time] wrote ${paths.md}`);
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
