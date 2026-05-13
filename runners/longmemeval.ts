/**
 * LongMemEval runner — budget edition.
 *
 * Runs the LongMemEval-S benchmark on a stratified sample (default n=50)
 * using Gemini 2.5 Flash as both the agent model and the judge. Tests three
 * conditions:
 *
 *   - baseline                no retrieval context (agent sees only the
 *                             question and its asked-on date)
 *   - continuity-blanket      BM25 retrieve on the question, top-K sessions
 *   - continuity-in-loop      entity-keyed BM25 (mirrors prod middleware),
 *                             top-K sessions
 *
 * NB: This is NOT a like-for-like LongMemEval leaderboard reproduction:
 *   - Baseline omits the full ~115k-token chat history (intentional, budget-
 *     constrained); we measure "does retrieval help?" not "does the model
 *     handle long context?".
 *   - Judge is Gemini 2.5 Flash, not the official GPT-4o judge. Re-judging
 *     a subsample with GPT-4o is the path to leaderboard-comparable numbers.
 *
 * Cost on 50 questions × 3 conditions × Gemini Flash: ~$0.10-0.30 if
 * outside the free tier; $0 with the daily free quota.
 *
 * Usage:
 *   GOOGLE_API_KEY=... npx tsx runners/longmemeval.ts \
 *     --sample fixtures/longmemeval/sample-50.json \
 *     --output reports/longmemeval/run-1 \
 *     --top-k 5
 *
 * Resume: re-run the same command. Checkpoint at
 *   <output>/results.checkpoint.json is consulted on startup and any
 *   (question_id, condition) tuples already completed are skipped.
 */
import * as fs from 'fs';
import * as path from 'path';
import type { Decision } from './shared/fixtures';
import { BM25Retriever, extractEntities, renderContext } from './shared/retrieval';

// ── Types ────────────────────────────────────────────────────────────────────

interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

interface LongMemEvalRecord {
  question_id: string;
  question_type: string;
  question: string;
  question_date: string;
  answer: string;
  answer_session_ids: string[];
  haystack_session_ids: string[];
  haystack_dates: string[];
  haystack_sessions: Turn[][];
}

type Condition = 'baseline' | 'continuity-blanket' | 'continuity-in-loop';

interface ResultRecord {
  question_id: string;
  question_type: string;
  condition: Condition;
  expected_answer: string;
  agent_response: string;
  autoeval_label: 0 | 1; // 1 = judge says correct
  judge_reasoning: string;
  judge_parse_ok: boolean;
  retrieved_session_ids?: string[];
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(argv: string[]) {
  const args: Record<string, string> = {};
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i];
    if (!t.startsWith('--')) continue;
    const eq = t.indexOf('=');
    if (eq !== -1) {
      args[t.slice(2, eq)] = t.slice(eq + 1);
    } else {
      args[t.slice(2)] = argv[i + 1];
      i++;
    }
  }
  return {
    sample: args.sample ?? 'fixtures/longmemeval/sample-50.json',
    output: args.output ?? 'reports/longmemeval/run-1',
    topK: parseInt(args['top-k'] ?? '5', 10),
    conditions: (args.conditions ?? 'baseline,continuity-blanket,continuity-in-loop').split(',') as Condition[],
    model: args.model ?? 'gemini-2.5-flash',
    sleepMs: parseInt(args['sleep-ms'] ?? '500', 10),
  };
}

// ── Session → Decision adapter ───────────────────────────────────────────────

function renderSession(session: Turn[]): string {
  return session
    .map((t) => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content}`)
    .join('\n');
}

function sessionsToDecisions(rec: LongMemEvalRecord): Decision[] {
  const out: Decision[] = [];
  for (let i = 0; i < rec.haystack_sessions.length; i++) {
    const session = rec.haystack_sessions[i];
    if (!session || session.length === 0) continue;
    const firstUser = session.find((t) => t.role === 'user');
    const title = firstUser ? firstUser.content.slice(0, 200) : `session ${i}`;
    out.push({
      id: `${rec.question_id}-s${i}`,
      question: title,
      answer: renderSession(session),
      tags: [rec.question_type],
      timestamp: rec.haystack_dates[i] ?? undefined,
      status: 'active',
    });
  }
  return out;
}

// ── Gemini client (inline; minimal — uses native fetch) ──────────────────────

const GEMINI_KEY = process.env.GOOGLE_API_KEY ?? loadEnvKey('GOOGLE_API_KEY');

function loadEnvKey(name: string): string {
  const envPath = path.resolve(__dirname, '..', '.env');
  if (!fs.existsSync(envPath)) return '';
  for (const line of fs.readFileSync(envPath, 'utf-8').split('\n')) {
    const m = line.match(/^([A-Z_]+)=(.+)$/);
    if (m && m[1] === name) return m[2].trim();
  }
  return '';
}

if (!GEMINI_KEY) {
  throw new Error('GOOGLE_API_KEY not set in env or .env');
}

async function geminiCall(prompt: string, opts: { model: string; maxRetries?: number } = { model: 'gemini-2.5-flash' }): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${opts.model}:generateContent?key=${GEMINI_KEY}`;
  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.0, maxOutputTokens: 1024 },
  };
  // Free tier limit is 15 RPM on gemini-2.5-flash — be generous with retry
  // backoff so transient 429s recover instead of dropping records.
  const maxRetries = opts.maxRetries ?? 6;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const errText = await resp.text();
        if (resp.status === 429 || resp.status >= 500) {
          // 5s, 10s, 15s, 20s, 25s, 30s — exponential-ish backoff for rate limits
          await sleep(5000 * (attempt + 1));
          continue;
        }
        throw new Error(`Gemini ${resp.status}: ${errText.slice(0, 200)}`);
      }
      const data = (await resp.json()) as { candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }> };
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
      return text;
    } catch (e) {
      if (attempt === maxRetries - 1) throw e;
      await sleep(5000 * (attempt + 1));
    }
  }
  throw new Error('gemini call exhausted retries');
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ── Prompts ──────────────────────────────────────────────────────────────────

function agentPrompt(rec: LongMemEvalRecord, contextBlock: string): string {
  const header = contextBlock
    ? `${contextBlock}\n\n---\n`
    : '';
  return `${header}Today is ${rec.question_date}.

A user asks you the following question. Use the retrieved conversation history (if any) above to answer. If the history doesn't contain enough information to answer reliably, say "I don't know."

Question: ${rec.question}

Answer concisely.`;
}

// LongMemEval-style autoeval prompt. NOT the official prompt — port from
// xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py later for leaderboard
// comparability. This is a faithful approximation.
function judgePrompt(rec: LongMemEvalRecord, response: string): string {
  return `You are evaluating whether a model's response correctly answers a question, given the ground-truth answer.

Question: ${rec.question}
Expected answer: ${rec.answer}
Model response: ${response}

Score:
- 1 if the model response is consistent with the expected answer (matches its key facts; "I don't know" only matches abstention questions where the expected answer is also "no clear answer")
- 0 otherwise

Respond in strict JSON: {"label": 0 or 1, "reasoning": "<one sentence>"}`;
}

function parseJudge(text: string): { label: 0 | 1; reasoning: string; ok: boolean } {
  // Strip code fences if present
  let s = text.trim();
  s = s.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();
  const m = s.match(/\{[\s\S]*\}/);
  if (!m) return { label: 0, reasoning: `unparseable: ${text.slice(0, 100)}`, ok: false };
  try {
    const j = JSON.parse(m[0]) as { label?: number; reasoning?: string };
    const label = j.label === 1 ? 1 : 0;
    return { label, reasoning: j.reasoning ?? '', ok: true };
  } catch {
    return { label: 0, reasoning: `JSON.parse failed: ${m[0].slice(0, 100)}`, ok: false };
  }
}

// ── Retrieval per condition ──────────────────────────────────────────────────

function retrieveForCondition(
  retriever: BM25Retriever,
  question: string,
  condition: Condition,
  topK: number,
): Decision[] {
  if (condition === 'baseline') return [];
  if (condition === 'continuity-blanket') return retriever.retrieve(question, topK);
  if (condition === 'continuity-in-loop') {
    const entityQuery = extractEntities(question);
    return retriever.retrieve(entityQuery, topK);
  }
  return [];
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv);
  const samplePath = path.resolve(args.sample);
  const outDir = path.resolve(args.output);
  fs.mkdirSync(outDir, { recursive: true });

  const records: LongMemEvalRecord[] = JSON.parse(fs.readFileSync(samplePath, 'utf-8'));
  console.log(`[longmemeval] loaded ${records.length} records from ${args.sample}`);
  console.log(`[longmemeval] conditions: ${args.conditions.join(', ')}`);
  console.log(`[longmemeval] model: ${args.model}  top-K: ${args.topK}`);

  const checkpointPath = path.join(outDir, 'results.checkpoint.json');
  let results: ResultRecord[] = [];
  let doneKeys = new Set<string>();
  if (fs.existsSync(checkpointPath)) {
    const ck = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8')) as { results: ResultRecord[] };
    results = ck.results;
    doneKeys = new Set(results.map((r) => `${r.question_id}::${r.condition}`));
    console.log(`[longmemeval] resuming from checkpoint: ${results.length} results already done`);
  }

  const totalUnits = records.length * args.conditions.length;
  let done = 0;
  const start = Date.now();

  for (const rec of records) {
    const decisions = sessionsToDecisions(rec);
    const retriever = new BM25Retriever(decisions);

    for (const cond of args.conditions) {
      done++;
      const key = `${rec.question_id}::${cond}`;
      if (doneKeys.has(key)) continue;

      const retrieved = retrieveForCondition(retriever, rec.question, cond, args.topK);
      const contextBlock = renderContext(retrieved);
      const ap = agentPrompt(rec, contextBlock);

      let agentResp = '';
      let judgeRaw = '';
      let judgeRes = { label: 0 as 0 | 1, reasoning: '', ok: false };
      let failed = false;
      try {
        agentResp = await geminiCall(ap, { model: args.model });
        await sleep(args.sleepMs);
        judgeRaw = await geminiCall(judgePrompt(rec, agentResp), { model: args.model });
        judgeRes = parseJudge(judgeRaw);
      } catch (e) {
        const err = e instanceof Error ? e.message : String(e);
        console.log(`  ! error on ${key}: ${err.slice(0, 120)} (will retry on resume)`);
        failed = true;
      }

      if (failed) {
        // Don't record this unit; leave it unfinished so a re-run picks it up.
        continue;
      }

      results.push({
        question_id: rec.question_id,
        question_type: rec.question_type,
        condition: cond,
        expected_answer: rec.answer,
        agent_response: agentResp,
        autoeval_label: judgeRes.label,
        judge_reasoning: judgeRes.reasoning,
        judge_parse_ok: judgeRes.ok,
        retrieved_session_ids: retrieved.map((d) => d.id),
      });
      doneKeys.add(key);

      if (results.length % 5 === 0) {
        fs.writeFileSync(checkpointPath, JSON.stringify({ results }, null, 2));
      }
      if (done <= 5 || done % 10 === 0) {
        const elapsed = (Date.now() - start) / 1000;
        const rate = done / Math.max(1, elapsed);
        const eta = (totalUnits - done) / Math.max(0.0001, rate);
        console.log(`  [${done}/${totalUnits}] elapsed ${elapsed.toFixed(0)}s eta ${eta.toFixed(0)}s | ${cond} ${rec.question_type} label=${judgeRes.label}`);
      }
      await sleep(args.sleepMs);
    }
  }

  // ── Save ─────────────────────────────────────────────────────────────────
  const outPath = path.join(outDir, 'results.json');
  fs.writeFileSync(outPath, JSON.stringify({
    generated_at: new Date().toISOString(),
    sample: args.sample,
    n_records: records.length,
    conditions: args.conditions,
    agent_model: args.model,
    judge_model: args.model,
    top_k: args.topK,
    note: 'Baseline = NO context (not full history). Judge = Gemini Flash (not official GPT-4o). See file header.',
    results,
  }, null, 2));
  console.log(`[longmemeval] wrote ${outPath} (${results.length} results)`);

  // ── Summary ──────────────────────────────────────────────────────────────
  const summary = summarize(results, args.conditions);
  const summaryPath = path.join(outDir, 'summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(outDir, 'summary.md'), renderSummaryMd(summary));
  console.log(`[longmemeval] wrote ${summaryPath}`);

  // Checkpoint cleanup
  if (fs.existsSync(checkpointPath)) fs.unlinkSync(checkpointPath);
}

interface Summary {
  by_condition: Record<string, { n: number; accuracy: number; parse_ok_rate: number }>;
  by_condition_and_type: Record<string, Record<string, { n: number; accuracy: number }>>;
  per_condition_deltas: Record<string, number>;
}

function summarize(results: ResultRecord[], conditions: Condition[]): Summary {
  const by_condition: Summary['by_condition'] = {};
  const by_condition_and_type: Summary['by_condition_and_type'] = {};
  for (const c of conditions) {
    const subset = results.filter((r) => r.condition === c);
    const acc = subset.length === 0 ? 0 : subset.reduce((s, r) => s + r.autoeval_label, 0) / subset.length;
    const parseOk = subset.length === 0 ? 0 : subset.filter((r) => r.judge_parse_ok).length / subset.length;
    by_condition[c] = { n: subset.length, accuracy: acc, parse_ok_rate: parseOk };
    by_condition_and_type[c] = {};
    const types = Array.from(new Set(subset.map((r) => r.question_type))).sort();
    for (const t of types) {
      const ss = subset.filter((r) => r.question_type === t);
      by_condition_and_type[c][t] = {
        n: ss.length,
        accuracy: ss.length === 0 ? 0 : ss.reduce((s, r) => s + r.autoeval_label, 0) / ss.length,
      };
    }
  }
  const baselineAcc = by_condition['baseline']?.accuracy ?? 0;
  const per_condition_deltas: Record<string, number> = {};
  for (const c of conditions) {
    if (c === 'baseline') continue;
    per_condition_deltas[`${c}_minus_baseline`] = (by_condition[c]?.accuracy ?? 0) - baselineAcc;
  }
  return { by_condition, by_condition_and_type, per_condition_deltas };
}

function renderSummaryMd(s: Summary): string {
  const lines: string[] = ['# LongMemEval-S subsample — summary', ''];
  lines.push('## Overall accuracy by condition', '');
  lines.push('| Condition | N | Accuracy | Judge parse-ok |');
  lines.push('|---|---|---|---|');
  for (const [c, v] of Object.entries(s.by_condition)) {
    lines.push(`| ${c} | ${v.n} | ${(v.accuracy * 100).toFixed(1)}% | ${(v.parse_ok_rate * 100).toFixed(1)}% |`);
  }
  lines.push('', '## Δ vs baseline', '');
  for (const [k, v] of Object.entries(s.per_condition_deltas)) {
    lines.push(`- ${k}: ${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)} percentage points`);
  }
  lines.push('', '## Per-condition × per-question-type accuracy', '');
  const conditions = Object.keys(s.by_condition_and_type);
  const types = Array.from(new Set(conditions.flatMap((c) => Object.keys(s.by_condition_and_type[c])))).sort();
  lines.push(`| Question type | ${conditions.join(' | ')} |`);
  lines.push(`|---|${conditions.map(() => '---').join('|')}|`);
  for (const t of types) {
    const row = conditions.map((c) => {
      const cell = s.by_condition_and_type[c]?.[t];
      return cell ? `${(cell.accuracy * 100).toFixed(1)}% (n=${cell.n})` : '—';
    });
    lines.push(`| ${t} | ${row.join(' | ')} |`);
  }
  return lines.join('\n');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
