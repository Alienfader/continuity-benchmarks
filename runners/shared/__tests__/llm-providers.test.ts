/**
 * Tests for the benchmark LLM client factory + mock client.
 *
 * Real OpenAI / Anthropic / Ollama calls are NOT exercised here — the spec
 * forbids making real API calls in unit tests. The networked clients are
 * covered indirectly by the end-to-end runner dry-run under mock mode.
 */

import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import {
  MockLLMClient,
  createLLMClient,
  OpenAIBenchmarkClient,
  AnthropicBenchmarkClient,
  OllamaBenchmarkClient,
} from '../llm-providers';

test('MockLLMClient echoes prompt and increments callIndex', async () => {
  const c = new MockLLMClient({ latencyMs: 0 });
  const r1 = await c.complete('first');
  const r2 = await c.complete('second');
  assert.match(r1.text, /\[mock-response-0\]/);
  assert.match(r2.text, /\[mock-response-1\]/);
  assert.equal(r1.model, 'mock');
  assert.equal(r1.provider, 'mock');
  assert.ok(r1.inputTokens > 0);
  assert.ok(r1.outputTokens > 0);
});

test('MockLLMClient uses custom responder', async () => {
  const c = new MockLLMClient({
    latencyMs: 0,
    responder: (p, i) => `call-${i}::${p}`,
  });
  const r = await c.complete('hi');
  assert.equal(r.text, 'call-0::hi');
});

test('createLLMClient(mock) returns MockLLMClient', () => {
  const c = createLLMClient('mock');
  assert.ok(c instanceof MockLLMClient);
  assert.equal(c.getModelName(), 'mock');
  assert.equal(c.getProviderName(), 'mock');
});

test('createLLMClient(gpt-4o-mini) requires an API key', () => {
  const orig = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  try {
    assert.throws(
      () => createLLMClient('gpt-4o-mini'),
      /OpenAI API key missing/,
    );
  } finally {
    if (orig !== undefined) process.env.OPENAI_API_KEY = orig;
  }
});

test('createLLMClient(gpt-4o-mini) constructs OpenAIBenchmarkClient with key', () => {
  const c = createLLMClient('gpt-4o-mini', { apiKey: 'sk-test' });
  assert.ok(c instanceof OpenAIBenchmarkClient);
  assert.equal(c.getModelName(), 'gpt-4o-mini');
  assert.equal(c.getProviderName(), 'openai');
});

test('createLLMClient(claude-sonnet-4-6) requires Anthropic key', () => {
  const orig = process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;
  try {
    assert.throws(() => createLLMClient('claude-sonnet-4-6'), /Anthropic API key missing/);
  } finally {
    if (orig !== undefined) process.env.ANTHROPIC_API_KEY = orig;
  }
});

test('createLLMClient(claude-sonnet-4-6) constructs AnthropicBenchmarkClient with key', () => {
  const c = createLLMClient('claude-sonnet-4-6', { apiKey: 'sk-test' });
  assert.ok(c instanceof AnthropicBenchmarkClient);
  assert.equal(c.getProviderName(), 'anthropic');
});

test('createLLMClient(qwen2.5-7b) constructs OllamaBenchmarkClient', () => {
  const c = createLLMClient('qwen2.5-7b');
  assert.ok(c instanceof OllamaBenchmarkClient);
  assert.equal(c.getProviderName(), 'ollama');
  assert.equal(c.getModelName(), 'qwen2.5:7b');
});

test('createLLMClient(unknown) throws', () => {
  // @ts-expect-error — intentional bad input to verify exhaustiveness
  assert.throws(() => createLLMClient('not-a-model'));
});
