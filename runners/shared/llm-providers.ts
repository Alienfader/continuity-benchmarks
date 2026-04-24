/**
 * Benchmark-only LLM client abstraction.
 *
 * The production LLMProviderManager is VS-Code-bound. These runners are
 * Node-only, so we talk to each provider via its REST API / SDK directly
 * and expose a single uniform `LLMClient` interface across:
 *
 *   - OpenAI     (gpt-4o, gpt-4o-mini)       via fetch()
 *   - Anthropic  (claude-sonnet-4.x)         via @anthropic-ai/sdk
 *   - Ollama     (qwen2.5:7b or similar)     via fetch() to localhost
 *   - Mock       (deterministic)             for unit tests / dry runs
 *
 * This mirrors ID-RAG's model lineup (GPT-4o, GPT-4o mini, Qwen2.5-7B)
 * plus Claude Sonnet for the alignment judge and convergence runner.
 */

import Anthropic from '@anthropic-ai/sdk';

export type SupportedModel =
  | 'gpt-4o'
  | 'gpt-4o-mini'
  | 'qwen2.5-7b'
  | 'claude-sonnet-4-5'
  | 'claude-sonnet-4-6'
  | 'mock';

export interface LLMCallOptions {
  systemPrompt?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface LLMCallResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
  model: string;
  provider: string;
  durationMs: number;
}

export interface LLMClient {
  complete(prompt: string, opts?: LLMCallOptions): Promise<LLMCallResult>;
  getModelName(): string;
  getProviderName(): string;
}

// ── Mock client (deterministic, no network) ──────────────────────────────────

export interface MockLLMClientOptions {
  /**
   * If provided, the mock returns `responder(prompt, callIndex)` for each call.
   * Otherwise it echoes the prompt back with a prefix.
   */
  responder?: (prompt: string, callIndex: number, opts?: LLMCallOptions) => string;
  /** Simulated per-call latency in ms. Default 1ms. */
  latencyMs?: number;
}

export class MockLLMClient implements LLMClient {
  private callIndex = 0;

  constructor(private readonly opts: MockLLMClientOptions = {}) {}

  async complete(prompt: string, opts?: LLMCallOptions): Promise<LLMCallResult> {
    const start = Date.now();
    const latency = this.opts.latencyMs ?? 1;
    if (latency > 0) await new Promise((r) => setTimeout(r, latency));

    const text = this.opts.responder
      ? this.opts.responder(prompt, this.callIndex, opts)
      : `[mock-response-${this.callIndex}] ${prompt.slice(0, 80)}`;
    this.callIndex += 1;

    return {
      text,
      inputTokens: Math.ceil(prompt.length / 4),
      outputTokens: Math.ceil(text.length / 4),
      model: 'mock',
      provider: 'mock',
      durationMs: Date.now() - start,
    };
  }

  getModelName(): string {
    return 'mock';
  }

  getProviderName(): string {
    return 'mock';
  }
}

// ── OpenAI client (gpt-4o, gpt-4o-mini) ──────────────────────────────────────

export class OpenAIBenchmarkClient implements LLMClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly model: string;

  constructor(model: 'gpt-4o' | 'gpt-4o-mini', opts: { apiKey?: string; baseUrl?: string } = {}) {
    const apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        'OpenAI API key missing. Set OPENAI_API_KEY or pass { apiKey } to the client.',
      );
    }
    this.apiKey = apiKey;
    this.baseUrl = opts.baseUrl ?? 'https://api.openai.com/v1';
    this.model = model;
  }

  async complete(prompt: string, opts: LLMCallOptions = {}): Promise<LLMCallResult> {
    const start = Date.now();
    const messages: Array<{ role: string; content: string }> = [];
    if (opts.systemPrompt) messages.push({ role: 'system', content: opts.systemPrompt });
    messages.push({ role: 'user', content: prompt });

    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages,
        max_tokens: opts.maxTokens ?? 1024,
        temperature: opts.temperature ?? 0.2,
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`OpenAI ${this.model} ${res.status}: ${body.slice(0, 300)}`);
    }

    const data = (await res.json()) as {
      choices: Array<{ message: { content: string } }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    const text = data.choices?.[0]?.message?.content ?? '';
    return {
      text,
      inputTokens: data.usage?.prompt_tokens ?? 0,
      outputTokens: data.usage?.completion_tokens ?? 0,
      model: this.model,
      provider: 'openai',
      durationMs: Date.now() - start,
    };
  }

  getModelName(): string {
    return this.model;
  }

  getProviderName(): string {
    return 'openai';
  }
}

// ── Anthropic client (claude-sonnet-*) ───────────────────────────────────────

export class AnthropicBenchmarkClient implements LLMClient {
  private readonly client: Anthropic;
  private readonly model: string;

  constructor(model: string, opts: { apiKey?: string } = {}) {
    const apiKey = opts.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error('Anthropic API key missing. Set ANTHROPIC_API_KEY.');
    }
    this.client = new Anthropic({ apiKey });
    this.model = model;
  }

  async complete(prompt: string, opts: LLMCallOptions = {}): Promise<LLMCallResult> {
    const start = Date.now();
    const resp = await this.client.messages.create({
      model: this.model,
      max_tokens: opts.maxTokens ?? 1024,
      temperature: opts.temperature ?? 0.2,
      system: opts.systemPrompt,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = resp.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('\n');

    return {
      text,
      inputTokens: resp.usage?.input_tokens ?? 0,
      outputTokens: resp.usage?.output_tokens ?? 0,
      model: this.model,
      provider: 'anthropic',
      durationMs: Date.now() - start,
    };
  }

  getModelName(): string {
    return this.model;
  }

  getProviderName(): string {
    return 'anthropic';
  }
}

// ── Ollama client (qwen2.5:7b) ───────────────────────────────────────────────

export class OllamaBenchmarkClient implements LLMClient {
  private readonly baseUrl: string;
  private readonly model: string;

  constructor(model: string, opts: { baseUrl?: string } = {}) {
    this.baseUrl = opts.baseUrl ?? process.env.OLLAMA_HOST ?? 'http://localhost:11434';
    this.model = model;
  }

  async complete(prompt: string, opts: LLMCallOptions = {}): Promise<LLMCallResult> {
    const start = Date.now();
    const fullPrompt = opts.systemPrompt
      ? `${opts.systemPrompt}\n\n${prompt}`
      : prompt;

    const res = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        prompt: fullPrompt,
        stream: false,
        options: {
          num_predict: opts.maxTokens ?? 1024,
          temperature: opts.temperature ?? 0.2,
        },
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Ollama ${this.model} ${res.status}: ${body.slice(0, 300)}`);
    }

    const data = (await res.json()) as {
      response?: string;
      prompt_eval_count?: number;
      eval_count?: number;
    };
    const text = data.response ?? '';
    return {
      text,
      inputTokens: data.prompt_eval_count ?? 0,
      outputTokens: data.eval_count ?? 0,
      model: this.model,
      provider: 'ollama',
      durationMs: Date.now() - start,
    };
  }

  getModelName(): string {
    return this.model;
  }

  getProviderName(): string {
    return 'ollama';
  }
}

// ── Factory ──────────────────────────────────────────────────────────────────

export interface CreateLLMClientOptions {
  apiKey?: string;
  baseUrl?: string;
  mock?: MockLLMClientOptions;
}

export function createLLMClient(
  model: SupportedModel,
  opts: CreateLLMClientOptions = {},
): LLMClient {
  if (model === 'mock') return new MockLLMClient(opts.mock);
  if (model === 'gpt-4o' || model === 'gpt-4o-mini') {
    return new OpenAIBenchmarkClient(model, { apiKey: opts.apiKey, baseUrl: opts.baseUrl });
  }
  if (model === 'qwen2.5-7b') {
    return new OllamaBenchmarkClient('qwen2.5:7b', { baseUrl: opts.baseUrl });
  }
  if (model === 'claude-sonnet-4-5' || model === 'claude-sonnet-4-6') {
    const resolved =
      model === 'claude-sonnet-4-6' ? 'claude-sonnet-4-6' : 'claude-sonnet-4-5';
    return new AnthropicBenchmarkClient(resolved, { apiKey: opts.apiKey });
  }
  const exhaustive: never = model;
  throw new Error(`Unknown model: ${String(exhaustive)}`);
}
