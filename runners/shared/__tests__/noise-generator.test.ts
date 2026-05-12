/**
 * Tests for the noise generator — deterministic, sizing, topic handling.
 *
 * Run with: npx ts-node --test benchmarks/src/id-rag-parallel/runners/shared/__tests__/noise-generator.test.ts
 */

import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import { generateNoise, countTokensApprox } from '../noise-generator';

test('countTokensApprox returns ceil(len/4)', () => {
  assert.equal(countTokensApprox(''), 0);
  assert.equal(countTokensApprox('abcd'), 1);
  assert.equal(countTokensApprox('abcde'), 2);
  assert.equal(countTokensApprox('x'.repeat(4000)), 1000);
});

test('generateNoise produces at-least-target-tokens output', () => {
  const text = generateNoise({ targetTokens: 5000, seed: 1 });
  const tokens = countTokensApprox(text);
  assert.ok(tokens >= 5000, `expected >= 5000 tokens, got ${tokens}`);
  // Within 10% of target (one paragraph overshoot is allowed)
  assert.ok(tokens <= 6000, `expected <= 6000 tokens, got ${tokens}`);
});

test('generateNoise is deterministic for the same seed', () => {
  const a = generateNoise({ targetTokens: 500, seed: 7 });
  const b = generateNoise({ targetTokens: 500, seed: 7 });
  assert.equal(a, b);
});

test('generateNoise differs between seeds', () => {
  const a = generateNoise({ targetTokens: 500, seed: 1 });
  const b = generateNoise({ targetTokens: 500, seed: 999 });
  assert.notEqual(a, b);
});

test('generateNoise supports each topic source', () => {
  for (const topic of ['wikipedia', 'stackoverflow', 'off-topic-decisions', 'mix'] as const) {
    const text = generateNoise({ targetTokens: 200, seed: 42, topic });
    assert.ok(text.length > 0, `topic=${topic} returned empty`);
  }
});

test('generateNoise wikipedia topic does NOT contain stackoverflow markers', () => {
  const text = generateNoise({ targetTokens: 500, seed: 42, topic: 'wikipedia' });
  assert.ok(!/^Q:/m.test(text), 'wikipedia topic leaked Q: lines');
});
