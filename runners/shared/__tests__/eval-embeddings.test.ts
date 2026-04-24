/**
 * Tests for the eval-embedder wrapper. Uses MockEmbedder — the real
 * all-mpnet-base-v2 path is exercised by the end-to-end runner dry-run, not
 * here, because pulling the model weights is slow and external.
 */

import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import {
  MockEmbedder,
  cosineSimilarity,
  scoreAnswer,
  summarizeScores,
} from '../eval-embeddings';

test('cosineSimilarity of identical vectors is 1.0', () => {
  const v = new Float32Array([0.5, 0.5, 0.5, 0.5]);
  assert.ok(Math.abs(cosineSimilarity(v, v) - 1) < 1e-6);
});

test('cosineSimilarity of orthogonal vectors is 0.0', () => {
  const a = new Float32Array([1, 0, 0, 0]);
  const b = new Float32Array([0, 1, 0, 0]);
  assert.ok(Math.abs(cosineSimilarity(a, b)) < 1e-6);
});

test('cosineSimilarity throws on dimension mismatch', () => {
  const a = new Float32Array([1, 0]);
  const b = new Float32Array([1, 0, 0]);
  assert.throws(() => cosineSimilarity(a, b));
});

test('cosineSimilarity handles zero vectors without NaN', () => {
  const a = new Float32Array([0, 0, 0]);
  const b = new Float32Array([1, 1, 1]);
  const s = cosineSimilarity(a, b);
  assert.ok(Number.isFinite(s));
  assert.equal(s, 0);
});

test('MockEmbedder: identical text scores ~1.0', async () => {
  const embedder = new MockEmbedder();
  const s = await scoreAnswer('hello world', 'hello world', embedder);
  assert.ok(s > 0.99, `expected ~1.0, got ${s}`);
});

test('MockEmbedder: different text scores below 1.0', async () => {
  const embedder = new MockEmbedder();
  const s = await scoreAnswer('hello world', 'completely unrelated', embedder);
  assert.ok(s < 0.99, `expected < 1, got ${s}`);
});

test('summarizeScores empty input', () => {
  const s = summarizeScores([]);
  assert.equal(s.count, 0);
  assert.equal(s.mean, 0);
  assert.equal(s.fractionAbove070, 0);
});

test('summarizeScores computes mean/median/min/max/fraction', () => {
  const s = summarizeScores([0.2, 0.9, 0.8, 0.5, 0.95]);
  assert.equal(s.count, 5);
  assert.ok(Math.abs(s.mean - 0.67) < 0.01);
  assert.equal(s.median, 0.8);
  assert.equal(s.min, 0.2);
  assert.equal(s.max, 0.95);
  // 0.9, 0.8, 0.95 are >= 0.7 → 3/5
  assert.ok(Math.abs(s.fractionAbove070 - 0.6) < 1e-9);
});
