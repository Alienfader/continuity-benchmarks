/**
 * Tool-calling agent client for `runners/middleware-replay.ts`.
 *
 * The base `LLMClient` in `llm-providers.ts` is text-in/text-out. For
 * the production-replay agent loop we need:
 *
 *   1. Turn 1: send a prompt + a tool advertisement; agent returns
 *      either a final text answer OR one-or-more tool-use calls.
 *   2. Tool dispatch: caller (the runner) routes the tool call to MCP
 *      and captures the result.
 *   3. Turn 2: send the prior conversation + tool result back; agent
 *      returns a final text answer.
 *
 * This module provides exactly that interface, with provider-specific
 * implementations for Anthropic, OpenAI, and a deterministic Mock.
 *
 * We deliberately keep the surface narrow:
 *   - Only one round of tool-use is supported (turn 1 → tool → turn 2).
 *     If the agent tries to call another tool in turn 2, we treat its
 *     text content as the final answer and ignore the second call.
 *   - Only string-typed tool inputs (any JSON object) are accepted.
 *
 * Both restrictions match the experimental design of the v2 matrix:
 * one search call, one final answer. Multi-step agentic search is a
 * future expansion.
 */

import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';

export interface ToolDef {
  name: string;
  description: string;
  /** JSON Schema for the tool input. */
  inputSchema: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  toolCallId: string;
  /** Plain-text content the tool returned. JSON should be stringified by the caller. */
  content: string;
  isError?: boolean;
}

export interface AgentTurnArgs {
  systemPrompt: string;
  userMessage: string;
  tools: ToolDef[];
  /** Maximum tokens for the agent response. */
  maxTokens?: number;
  temperature?: number;
}

export interface AgentTurnResult {
  /** Plain-text content the agent emitted. May be '' if it called a tool with no commentary. */
  text: string;
  /** Tool calls the agent issued in this turn. Empty if the agent answered directly. */
  toolCalls: ToolCall[];
  inputTokens: number;
  outputTokens: number;
  /** Provider-specific assistant message blob — needed verbatim by continueWithToolResults to preserve ordering. */
  rawAssistantBlob: unknown;
}

export interface AgentContinueArgs {
  systemPrompt: string;
  userMessage: string;
  tools: ToolDef[];
  /** The assistant blob from the prior turn — caller must pass through unchanged. */
  priorAssistantBlob: unknown;
  /** Tool results matching the priorTurn's toolCalls (one ToolResult per ToolCall). */
  toolResults: ToolResult[];
  maxTokens?: number;
  temperature?: number;
}

export interface ToolCallingAgent {
  decideToolCall(args: AgentTurnArgs): Promise<AgentTurnResult>;
  continueWithToolResults(args: AgentContinueArgs): Promise<AgentTurnResult>;
  getModelName(): string;
  getProviderName(): string;
}

// ────────────────────────────────────────────────────────────────────
// Anthropic
// ────────────────────────────────────────────────────────────────────

export class AnthropicAgent implements ToolCallingAgent {
  private readonly client: Anthropic;
  private readonly modelName: string;

  constructor(modelName: string, opts: { apiKey?: string } = {}) {
    const apiKey = opts.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('AnthropicAgent: ANTHROPIC_API_KEY is required');
    this.client = new Anthropic({ apiKey });
    this.modelName = modelName;
  }

  async decideToolCall(args: AgentTurnArgs): Promise<AgentTurnResult> {
    const response = await this.client.messages.create({
      model: this.modelName,
      max_tokens: args.maxTokens ?? 1024,
      temperature: args.temperature ?? 0.2,
      system: args.systemPrompt,
      tools: args.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema as Anthropic.Messages.Tool.InputSchema,
      })),
      messages: [{ role: 'user', content: args.userMessage }],
    });

    const text = response.content
      .filter((b): b is Anthropic.Messages.TextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('');

    const toolCalls: ToolCall[] = response.content
      .filter((b): b is Anthropic.Messages.ToolUseBlock => b.type === 'tool_use')
      .map((b) => ({
        id: b.id,
        name: b.name,
        arguments: b.input as Record<string, unknown>,
      }));

    return {
      text,
      toolCalls,
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      rawAssistantBlob: response.content,
    };
  }

  async continueWithToolResults(args: AgentContinueArgs): Promise<AgentTurnResult> {
    // Force the model to produce a final text answer in turn 2 by
    // appending an explicit instruction. We also drop the `tools` array
    // from this call — without tools advertised, the model has no
    // choice but to answer in text. (Anthropic doesn't support a
    // `tool_choice: "none"` analog, so omitting tools is the cleanest
    // path; the prior turn's tool_use blocks remain valid because they
    // exist in the assistant message, not in the current request's
    // tools array.)
    const closingNudge =
      "Now produce the final answer based on the tool result above. Do not call additional tools.";
    const response = await this.client.messages.create({
      model: this.modelName,
      max_tokens: args.maxTokens ?? 1024,
      temperature: args.temperature ?? 0.2,
      system: args.systemPrompt,
      messages: [
        { role: 'user', content: args.userMessage },
        { role: 'assistant', content: args.priorAssistantBlob as Anthropic.Messages.ContentBlock[] },
        {
          role: 'user',
          content: [
            ...args.toolResults.map((r) => ({
              type: 'tool_result' as const,
              tool_use_id: r.toolCallId,
              content: r.content,
              is_error: r.isError,
            })),
            { type: 'text' as const, text: closingNudge },
          ],
        },
      ],
    });

    const text = response.content
      .filter((b): b is Anthropic.Messages.TextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('');
    return {
      text,
      toolCalls: [],
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      rawAssistantBlob: response.content,
    };
  }

  getModelName(): string {
    return this.modelName;
  }
  getProviderName(): string {
    return 'anthropic';
  }
}

// ────────────────────────────────────────────────────────────────────
// OpenAI (minimal, hand-rolled fetch — keeps this module SDK-agnostic
// across versions; OpenAI's tool-use API is stable).
// ────────────────────────────────────────────────────────────────────

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OpenAIChatChoice {
  message: {
    role: string;
    content: string | null;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason: string;
}

interface OpenAIChatResponse {
  choices: OpenAIChatChoice[];
  usage: { prompt_tokens: number; completion_tokens: number };
}

export class OpenAIAgent implements ToolCallingAgent {
  private readonly apiKey: string;
  private readonly modelName: string;
  private readonly baseUrl: string;

  constructor(modelName: string, opts: { apiKey?: string; baseUrl?: string } = {}) {
    const apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OpenAIAgent: OPENAI_API_KEY is required');
    this.apiKey = apiKey;
    this.modelName = modelName;
    this.baseUrl = opts.baseUrl ?? 'https://api.openai.com/v1';
  }

  async decideToolCall(args: AgentTurnArgs): Promise<AgentTurnResult> {
    const body = {
      model: this.modelName,
      messages: [
        { role: 'system', content: args.systemPrompt },
        { role: 'user', content: args.userMessage },
      ],
      tools: args.tools.map((t) => ({
        type: 'function' as const,
        function: {
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        },
      })),
      // Force serial tool use. Our 2-turn protocol expects one tool
      // call per turn 1; multiple parallel tool_calls would require us
      // to dispatch and respond to ALL of them in turn 2 before the
      // model will accept the messages array.
      parallel_tool_calls: false,
      max_tokens: args.maxTokens ?? 1024,
      temperature: args.temperature ?? 0.2,
    };

    const response = await this.callOpenAI(body);
    const choice = response.choices[0];
    const text = choice.message.content ?? '';
    const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map((c) => ({
      id: c.id,
      name: c.function.name,
      arguments: safeParse(c.function.arguments),
    }));

    return {
      text,
      toolCalls,
      inputTokens: response.usage.prompt_tokens,
      outputTokens: response.usage.completion_tokens,
      rawAssistantBlob: choice.message,
    };
  }

  async continueWithToolResults(args: AgentContinueArgs): Promise<AgentTurnResult> {
    const priorMsg = args.priorAssistantBlob as OpenAIChatChoice['message'];
    // Force a final text answer in turn 2: tool_choice=none disables
    // additional tool calls, and an inline nudge in the trailing user
    // message tells the model to answer based on the tool result. We
    // still pass `tools` because some OpenAI models reject
    // tool_choice=none without it.
    const closingNudge =
      "Now produce the final answer based on the tool result above. Do not call additional tools.";
    const body = {
      model: this.modelName,
      messages: [
        { role: 'system', content: args.systemPrompt },
        { role: 'user', content: args.userMessage },
        priorMsg,
        ...args.toolResults.map((r) => ({
          role: 'tool' as const,
          tool_call_id: r.toolCallId,
          content: r.content,
        })),
        { role: 'user', content: closingNudge },
      ],
      tools: args.tools.map((t) => ({
        type: 'function' as const,
        function: {
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        },
      })),
      tool_choice: 'none' as const,
      max_tokens: args.maxTokens ?? 1024,
      temperature: args.temperature ?? 0.2,
    };

    const response = await this.callOpenAI(body);
    const choice = response.choices[0];
    return {
      text: choice.message.content ?? '',
      toolCalls: [],
      inputTokens: response.usage.prompt_tokens,
      outputTokens: response.usage.completion_tokens,
      rawAssistantBlob: choice.message,
    };
  }

  private async callOpenAI(body: unknown): Promise<OpenAIChatResponse> {
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`OpenAI ${res.status}: ${text}`);
    }
    return (await res.json()) as OpenAIChatResponse;
  }

  getModelName(): string {
    return this.modelName;
  }
  getProviderName(): string {
    return 'openai';
  }
}

// ────────────────────────────────────────────────────────────────────
// Mock (deterministic, for smoke tests)
// ────────────────────────────────────────────────────────────────────

export interface MockAgentBehavior {
  /**
   * Called on turn 1. Return either a tool call or a final answer.
   * Default: issues a search_decisions call with the user prompt as the query.
   */
  turn1?: (args: AgentTurnArgs) => { text: string; toolCalls: ToolCall[] };
  /**
   * Called on turn 2 (after tool result). Default: echoes the first
   * tool result's content as the final answer.
   */
  turn2?: (args: AgentContinueArgs) => string;
}

export class MockAgent implements ToolCallingAgent {
  private callIndex = 0;
  constructor(private readonly behavior: MockAgentBehavior = {}) {}

  async decideToolCall(args: AgentTurnArgs): Promise<AgentTurnResult> {
    const fn =
      this.behavior.turn1 ??
      ((a: AgentTurnArgs) => {
        if (a.tools.length === 0) {
          return { text: `[mock-direct] ${a.userMessage.slice(-80)}`, toolCalls: [] };
        }
        const tool = a.tools[0];
        return {
          text: '',
          toolCalls: [
            {
              id: `mock-call-${this.callIndex++}`,
              name: tool.name,
              arguments:
                tool.name === 'search_decisions'
                  ? { query: a.userMessage, limit: 5 }
                  : tool.name === 'bash'
                  ? { command: `cat ${pickFixturePath(a.userMessage)}` }
                  : {},
            },
          ],
        };
      });
    const decision = fn(args);
    return {
      ...decision,
      inputTokens: Math.ceil(args.userMessage.length / 4),
      outputTokens: 16,
      rawAssistantBlob: { mock: true, ...decision },
    };
  }

  async continueWithToolResults(args: AgentContinueArgs): Promise<AgentTurnResult> {
    const fn =
      this.behavior.turn2 ??
      ((a: AgentContinueArgs) => {
        const first = a.toolResults[0];
        if (!first) return `[mock-final] no-tool-result for "${a.userMessage.slice(-60)}"`;
        return `[mock-final] ${first.content.slice(0, 240)}`;
      });
    const text = fn(args);
    return {
      text,
      toolCalls: [],
      inputTokens: Math.ceil(args.userMessage.length / 4) + Math.ceil((args.toolResults[0]?.content ?? '').length / 4),
      outputTokens: Math.ceil(text.length / 4),
      rawAssistantBlob: { mock: true, text },
    };
  }

  getModelName(): string {
    return 'mock';
  }
  getProviderName(): string {
    return 'mock';
  }
}

// ────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────

export function createAgent(model: string, opts: { mock?: MockAgentBehavior } = {}): ToolCallingAgent {
  if (model === 'mock') return new MockAgent(opts.mock);
  if (model === 'gpt-4o' || model === 'gpt-4o-mini') return new OpenAIAgent(model);
  if (model === 'claude-sonnet-4-5' || model === 'claude-sonnet-4-6') return new AnthropicAgent(model);
  throw new Error(`createAgent: unsupported model ${model}`);
}

// ────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────

function safeParse(jsonText: string): Record<string, unknown> {
  try {
    return JSON.parse(jsonText) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/**
 * Mock-only heuristic: pick a plausible fixture path to `cat` based on
 * keywords in the prompt. Only used by the deterministic MockAgent for
 * smoke-testing the auto-middleware wiring; real model agents pick paths
 * from their own reasoning. The keyword→path map mirrors the
 * code-links.json shipped in fixtures/data-pipeline/, so the mock
 * exercises the code-links lookup end-to-end on smoke runs.
 */
function pickFixturePath(prompt: string): string {
  const p = prompt.toLowerCase();
  const map: Array<[RegExp, string]> = [
    [/kafka|msk|kinesis/, 'src/streaming/kafka.ts'],
    [/schema\s*registry|avro|protobuf|serialization/, 'src/streaming/schema-registry.ts'],
    [/cdc|debezium/, 'src/streaming/cdc-debezium.ts'],
    [/dagster|airflow|orchestrat|kubernetes|k8s/, 'src/orchestration/dagster_pipeline.py'],
    [/snowflake|warehouse|bigquery|redshift/, 'warehouse/snowflake/connection.py'],
    [/dbt|transform/, 'warehouse/dbt/dbt_project.yml'],
    [/airbyte|connector/, 'src/connectors/airbyte_config.yml'],
    [/parquet|s3|partition|hive/, 'src/storage/s3_parquet.py'],
    [/glue|catalog/, 'src/catalog/glue_setup.py'],
    [/great\s*expectations|data\s*quality|soda/, 'src/dq/great_expectations_suite.py'],
    [/terraform|iac|infrastructure/, 'infra/terraform/main.tf'],
    [/secret|asm/, 'infra/secrets/asm.tf'],
    [/pii|tokeniz/, 'src/pii/tokenize.py'],
    [/python|mypy|pyproject/, 'pyproject.toml'],
  ];
  for (const [re, path] of map) {
    if (re.test(p)) return path;
  }
  return 'src/streaming/kafka.ts'; // default fallback that exists in code-links
}
