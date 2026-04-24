/**
 * Evaluation embeddings — cosine-similarity scoring for recall-over-time.
 *
 * Uses all-mpnet-base-v2 (via @xenova/transformers) for *evaluation only*.
 * Production retrieval uses MiniLM-L6-v2; keeping the eval model separate
 * avoids the "scored by the same model that retrieved" conflict ID-RAG
 * warns about.
 *
 * The embedder lazy-loads on first use (~400MB model download, cached in
 * ~/.cache/transformers/). For unit tests, pass a MockEmbedder instead.
 */

export interface Embedder {
  init(): Promise<void>;
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<Float32Array[]>;
  getModelName(): string;
}

// ── Real embedder (lazy-loaded all-mpnet-base-v2) ────────────────────────────

export class EvalEmbedder implements Embedder {
  private pipelineInstance: unknown = null;
  private initPromise: Promise<void> | null = null;
  private readonly modelId: string;

  constructor(modelId: string = 'Xenova/all-mpnet-base-v2') {
    this.modelId = modelId;
  }

  async init(): Promise<void> {
    if (this.pipelineInstance) return;
    if (!this.initPromise) {
      this.initPromise = (async () => {
        const { pipeline } = await import('@xenova/transformers');
        this.pipelineInstance = await pipeline('feature-extraction', this.modelId);
      })();
    }
    await this.initPromise;
  }

  async embed(text: string): Promise<Float32Array> {
    await this.init();
    const fn = this.pipelineInstance as (
      text: string | string[],
      opts: { pooling: 'mean'; normalize: boolean },
    ) => Promise<{ data: Float32Array }>;
    const output = await fn(text, { pooling: 'mean', normalize: true });
    return output.data;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (const t of texts) out.push(await this.embed(t));
    return out;
  }

  getModelName(): string {
    return this.modelId;
  }
}

// ── Mock embedder for unit tests ─────────────────────────────────────────────

/**
 * Deterministic hash-based embedder. Not semantic — for unit tests of the
 * scoring math only. Two identical strings yield identical vectors (score 1.0);
 * different strings yield low but non-zero similarity.
 */
export class MockEmbedder implements Embedder {
  constructor(private readonly dim: number = 32) {}

  async init(): Promise<void> {
    // no-op
  }

  async embed(text: string): Promise<Float32Array> {
    const vec = new Float32Array(this.dim);
    // Seeded per-character hash distributed across dims
    for (let i = 0; i < text.length; i++) {
      const c = text.charCodeAt(i);
      vec[(c + i) % this.dim] += Math.sin(c * (i + 1)) + 0.1;
    }
    // L2-normalize so cosine similarity is in [-1, 1]
    let norm = 0;
    for (let i = 0; i < this.dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < this.dim; i++) vec[i] /= norm;
    return vec;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }

  getModelName(): string {
    return 'mock-hash-embedder';
  }
}

// ── Math ─────────────────────────────────────────────────────────────────────

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Embedding dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Score a predicted answer against a ground-truth string.
 * Returns cosine similarity in [-1, 1]; typical "correct" answers land > 0.7
 * under all-mpnet-base-v2.
 */
export async function scoreAnswer(
  predicted: string,
  groundTruth: string,
  embedder: Embedder,
): Promise<number> {
  const [pVec, gVec] = await embedder.embedBatch([predicted, groundTruth]);
  return cosineSimilarity(pVec, gVec);
}

/**
 * Aggregate per-question scores into a recall summary.
 */
export interface RecallSummary {
  mean: number;
  median: number;
  min: number;
  max: number;
  count: number;
  /** Fraction of answers with similarity >= 0.7. */
  fractionAbove070: number;
}

export function summarizeScores(scores: number[]): RecallSummary {
  if (scores.length === 0) {
    return { mean: 0, median: 0, min: 0, max: 0, count: 0, fractionAbove070: 0 };
  }
  const sorted = [...scores].sort((a, b) => a - b);
  const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
  const median =
    sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[(sorted.length - 1) / 2];
  const above = scores.filter((s) => s >= 0.7).length;
  return {
    mean,
    median,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    count: scores.length,
    fractionAbove070: above / scores.length,
  };
}
