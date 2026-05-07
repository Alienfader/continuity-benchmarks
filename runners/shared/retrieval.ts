/**
 * Lightweight decision retrieval for the conditions runners compare. There
 * are now four named conditions; each runner uses the subset that makes
 * sense for its protocol.
 *
 *   - baseline:                       no decision context
 *
 *   - continuity:                     full-prompt-keyed retrieval, one-shot.
 *                                     Used by action-alignment as the "passive
 *                                     RAG / single retrieval on the user
 *                                     intent" condition. Used by recall-over-
 *                                     time as a backwards-compat alias for
 *                                     continuity-blanket (see below).
 *
 *   - continuity-blanket              recall-over-time only. Retrieve once
 *     (= legacy continuity for         using the concatenation of all 20
 *     recall-over-time)                quiz-question stems as the seed query;
 *                                     prepend that single context block to
 *                                     every session/question. Models a
 *                                     project-level "here's the context"
 *                                     blob delivered at session start.
 *
 *   - continuity-perq-frontloaded     recall-over-time only (M2 ablation
 *                                     comparand). Per-question retrieval
 *                                     computed ONCE at session 1; the same
 *                                     20 question-specific blobs are
 *                                     re-injected at every session boundary
 *                                     unchanged. Isolates "specific keying"
 *                                     from "fresh re-retrieval each session."
 *
 *   - continuity-in-loop:             For action-alignment: retrieval keyed
 *                                     on entities extracted from the prompt
 *                                     (file paths, capitalized nouns, tech
 *                                     terms) — mirrors what the production
 *                                     middleware does when it sees
 *                                     `edit_file(path="src/auth.ts")` and
 *                                     pulls decisions linked to that path.
 *                                     For recall-over-time: per-question
 *                                     retrieval done FRESH at every session
 *                                     boundary (re-fires under noise drift).
 *
 * Pairwise contrasts a reviewer can read from a 4-condition recall run:
 *   - baseline → blanket           : effect of any retrieval
 *   - blanket → perq-frontloaded   : effect of better retrieval keying
 *                                    (timing held constant)
 *   - perq-frontloaded → in-loop   : effect of fresh re-retrieval / timing
 *                                    (retrieval data held constant)
 *
 * For action-alignment (single-shot, no temporal dimension), the contrast
 * `continuity → continuity-in-loop` isolates the keying effect on action
 * correctness, holding the single-call protocol constant.
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

export type Condition =
  | 'baseline'
  | 'continuity'
  | 'continuity-blanket'
  | 'continuity-perq-frontloaded'
  | 'continuity-in-loop';

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

/**
 * Extract entity-shaped tokens from a prompt for middleware-style targeted
 * retrieval. Mirrors what the production AutoRetrievalMiddleware does when
 * it sees a tool call like `edit_file(path="src/auth.ts")` — it doesn't
 * BM25-search on the user's full intent string, it pulls file paths +
 * specific identifiers out of the tool args and queries on those.
 *
 * Three buckets, in order of specificity:
 *   1. Quoted strings (often file paths or specific terms in single/double quotes)
 *   2. File-path-shaped tokens (`src/auth.ts`, `migrations/0042.sql`, etc.)
 *   3. Capitalized identifiers ≥ 3 chars (Kafka, Postgres, GPT-4o, MSK, RDS)
 *
 * Falls back to the full prompt if nothing extractable — guarantees the
 * runner doesn't silently retrieve on an empty query.
 */
export function extractEntities(prompt: string): string {
  const entities: string[] = [];

  // 1. Quoted strings (single + double + backtick)
  const quoted = prompt.match(/["'`]([^"'`\n]+)["'`]/g) ?? [];
  for (const q of quoted) {
    entities.push(q.slice(1, -1));
  }

  // 2. File-path-shaped tokens (e.g. src/auth.ts, .continuity/decisions.json)
  const paths = prompt.match(/[a-zA-Z][a-zA-Z0-9_./-]*\.(ts|tsx|js|jsx|py|sql|json|yaml|yml|md|sh|rb|go|rs|java)/g) ?? [];
  entities.push(...paths);

  // 3. Capitalized identifiers (proper nouns / tech names: Kafka, MSK, Postgres, GPT-4o)
  // Match: starts with capital letter, ≥ 3 chars, may include digits/hyphens
  const caps = prompt.match(/\b[A-Z][A-Za-z0-9-]{2,}\b/g) ?? [];
  entities.push(...caps);

  if (entities.length === 0) return prompt;
  // Deduplicate while preserving order
  const seen = new Set<string>();
  const unique = entities.filter((e) => {
    const key = e.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return unique.join(' ');
}
