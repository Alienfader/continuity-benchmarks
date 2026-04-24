/**
 * Lightweight decision retrieval for the three conditions a runner compares:
 *
 *   - baseline:                no decision context
 *   - continuity:              top-K most relevant decisions for the question
 *                              (one-shot retrieval, like `get_quick_context`)
 *   - continuity-in-loop:      continuity + a retrieval step re-fires every
 *                              N agent turns (simulates Apollo's auto-retrieve-
 *                              before-tool-call middleware)
 *
 * We deliberately use a simple BM25-like TF-IDF scorer here rather than
 * importing SemanticSearchService. Three reasons:
 *   1. SemanticSearchService needs a writable .continuity dir + MiniLM model;
 *      it is heavy to spin up per-runner-per-question.
 *   2. A transparent, inspectable retriever makes the benchmark reviewable.
 *   3. The ID-RAG paper itself uses a simple bag-of-words retriever for its
 *      baseline "Continuity" condition, so this matches the paper's posture.
 *
 * Real-world runs with `--retriever=semantic` can later swap this out; the
 * interface is isolated so the runners don't care.
 */

import type { Decision } from './fixtures';

export type Condition = 'baseline' | 'continuity' | 'continuity-in-loop';

export interface Retriever {
  retrieve(query: string, k: number): Decision[];
}

const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'as', 'we', 'our',
  'why', 'what', 'how', 'did', 'do', 'does', 'use', 'using',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s\-_]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

export class BM25Retriever implements Retriever {
  private readonly docs: Decision[];
  private readonly docTokens: string[][];
  private readonly avgDocLen: number;
  private readonly docFreq: Map<string, number>;
  private readonly k1 = 1.2;
  private readonly b = 0.75;

  constructor(decisions: Decision[]) {
    this.docs = decisions;
    this.docTokens = decisions.map((d) => tokenize(`${d.question} ${d.answer} ${(d.tags ?? []).join(' ')}`));
    this.avgDocLen = this.docTokens.reduce((s, t) => s + t.length, 0) / Math.max(1, this.docTokens.length);
    this.docFreq = new Map();
    for (const tokens of this.docTokens) {
      const seen = new Set(tokens);
      for (const tok of seen) {
        this.docFreq.set(tok, (this.docFreq.get(tok) ?? 0) + 1);
      }
    }
  }

  retrieve(query: string, k: number): Decision[] {
    if (this.docs.length === 0) return [];
    const queryTokens = tokenize(query);
    if (queryTokens.length === 0) return [];
    const N = this.docs.length;
    const scores: Array<{ idx: number; score: number }> = [];
    for (let i = 0; i < this.docs.length; i++) {
      const docTokens = this.docTokens[i];
      const docLen = docTokens.length;
      if (docLen === 0) {
        scores.push({ idx: i, score: 0 });
        continue;
      }
      let score = 0;
      const tf = new Map<string, number>();
      for (const tok of docTokens) tf.set(tok, (tf.get(tok) ?? 0) + 1);
      for (const qTok of queryTokens) {
        const f = tf.get(qTok) ?? 0;
        if (f === 0) continue;
        const df = this.docFreq.get(qTok) ?? 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        const tfNorm = (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * docLen) / this.avgDocLen));
        score += idf * tfNorm;
      }
      // Prefer active decisions over superseded/outdated ones (simulates
      // `status: 'active'` filtering in the real handlers).
      const status = this.docs[i].status;
      if (status === 'superseded' || status === 'outdated' || status === 'deprecated') {
        score *= 0.4;
      }
      scores.push({ idx: i, score });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores
      .slice(0, k)
      .filter((s) => s.score > 0)
      .map((s) => this.docs[s.idx]);
  }
}

/**
 * Render a list of retrieved decisions as a Continuity-style context block.
 * Prepended to the agent prompt in the continuity + continuity-in-loop
 * conditions.
 */
export function renderContext(decisions: Decision[]): string {
  if (decisions.length === 0) return '';
  const lines: string[] = ['## Project decisions (retrieved from Continuity)'];
  for (const d of decisions) {
    const tagLine = d.tags && d.tags.length > 0 ? ` [${d.tags.join(', ')}]` : '';
    const status = d.status && d.status !== 'active' ? ` (status: ${d.status})` : '';
    lines.push('');
    lines.push(`### ${d.question}${status}${tagLine}`);
    lines.push(d.answer);
  }
  return lines.join('\n');
}
