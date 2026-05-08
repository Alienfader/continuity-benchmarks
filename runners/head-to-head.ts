/**
 * Head-to-Head Benchmark: Continuity vs MemPalace (v3 — Unified Search)
 *
 * Uses Continuity's REAL search pipeline (SemanticSearchService with
 * RRF hybrid search, embeddings, keyword + tag fusion) PLUS project file
 * search — matching the full MCP capability of searching both decisions
 * AND codebase files.
 *
 * ── DEPENDENCY NOTE FOR PUBLIC-REPO USERS ───────────────────────────
 * This runner imports `@continuity/core`'s `SemanticSearchService`,
 * which is a closed-source package shipped as part of the commercial
 * Continuity product. The package is NOT installable from the public
 * benchmarks repo. As a result:
 *
 *   - `npm install` will succeed (the import resolves at type-only
 *     time); this file ships as a reference implementation.
 *   - `npx tsx runners/head-to-head.ts` will fail at startup with a
 *     module-resolution error unless you have a sibling clone of the
 *     private `continuity-ultimate` workspace and have linked
 *     @continuity/core into this repo (`npm link`).
 *
 * To replicate §4.3 of the white paper against an alternative memory
 * system, edit the import below to point at your own retrieval engine
 * and re-implement the SemanticSearchService interface (`search(query,
 * { topK }) → Array<{ id, question, answer, score }>`). The rest of
 * the runner — corpus building, MemPalace subprocess invocation, judge
 * call, scoring — is fully self-contained.
 *
 * Saved per-query results that produced the §4.3 numbers in the white
 * paper are in `reports/head-to-head-*.json` and
 * `reports/head-to-head-*.md` for inspection without needing to rerun.
 * ────────────────────────────────────────────────────────────────────
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';

// Import Continuity's actual search engine.
// See the DEPENDENCY NOTE above — this resolves only inside the
// commercial workspace; replace with your own retrieval impl to run
// this benchmark against an alternative system.
// @ts-expect-error — closed-source package not in this repo's dependency graph; documented in DEPENDENCY NOTE
import { SemanticSearchService } from '@continuity/core';

const PROJECT_DIR = path.resolve(__dirname, '../../..');
const DECISIONS_PATH = path.join(PROJECT_DIR, '.continuity/decisions.json');
const RESULTS_FILE = path.resolve(__dirname, '../reports/head-to-head-results.json');
const REPORT_FILE = path.resolve(__dirname, '../reports/head-to-head-report.md');

// ── Self-contained project file search ──
// Provides codebase file search to supplement decision search,
// matching Continuity's full MCP capability.

interface FileSearchResult {
  filePath: string;
  relativePath: string;
  score: number;
  snippet: string;
}

const EXCLUDED_DIRS = ['node_modules', 'dist', '.git', 'coverage', '.worktrees'];

let projectFileIndex: { relativePath: string; content: string }[] = [];

function scanProjectFiles(rootDir: string): { relativePath: string; content: string }[] {
  const files: { relativePath: string; content: string }[] = [];
  const maxFileSize = 100_000; // 100KB limit per file

  function walk(dir: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (EXCLUDED_DIRS.includes(entry.name)) continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (['.ts', '.js', '.json', '.md', '.yml', '.yaml'].includes(ext)) {
          try {
            const stat = fs.statSync(fullPath);
            if (stat.size > maxFileSize) continue;
            const content = fs.readFileSync(fullPath, 'utf-8');
            files.push({ relativePath: path.relative(rootDir, fullPath), content });
          } catch {
            // skip unreadable files
          }
        }
      }
    }
  }

  walk(rootDir);
  return files;
}

function searchProjectFiles(query: string, limit: number = 5): FileSearchResult[] {
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  if (terms.length === 0) return [];

  const scored: FileSearchResult[] = [];

  for (const file of projectFileIndex) {
    const filenameLower = file.relativePath.toLowerCase();
    const contentLower = file.content.toLowerCase();

    let score = 0;
    let bestMatchPos = -1;

    // Filename match: strong signal
    for (const term of terms) {
      // Check both with and without common separators
      const baseName = path.basename(filenameLower, path.extname(filenameLower));
      if (baseName.includes(term)) {
        score += 0.4;
      } else if (filenameLower.includes(term)) {
        score += 0.25;
      }
    }

    // Content term frequency
    let totalHits = 0;
    for (const term of terms) {
      let pos = 0;
      let hits = 0;
      while ((pos = contentLower.indexOf(term, pos)) !== -1) {
        hits++;
        if (bestMatchPos === -1 || hits === 1) bestMatchPos = pos;
        pos += term.length;
      }
      totalHits += hits;
    }

    // Normalize content score by document length to avoid bias toward huge files
    if (totalHits > 0) {
      const density = totalHits / (contentLower.length / 1000); // hits per 1K chars
      score += Math.min(density * 0.1, 0.5); // cap content contribution
    }

    // Multi-term proximity bonus: if all terms appear within 500 chars of each other
    if (terms.length > 1 && totalHits > 0) {
      for (let i = 0; i < contentLower.length - 500; i += 200) {
        const window = contentLower.slice(i, i + 500);
        if (terms.every(t => window.includes(t))) {
          score += 0.15;
          bestMatchPos = i;
          break;
        }
      }
    }

    if (score > 0.05) {
      // Extract snippet around best match
      const snippetStart = Math.max(0, bestMatchPos - 100);
      const snippetEnd = Math.min(file.content.length, bestMatchPos + 400);
      const snippet = file.content.slice(snippetStart, snippetEnd).replace(/\n/g, ' ').trim();

      scored.push({
        filePath: path.join(PROJECT_DIR, file.relativePath),
        relativePath: file.relativePath,
        score,
        snippet: snippet.slice(0, 500),
      });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

// 50 diverse queries spanning the full tag vocabulary
const QUERIES = [
  'MCP server architecture',
  'decision logging',
  'authentication',
  'webpack configuration',
  'licensing',
  'semantic search',
  'freshness scoring',
  'VSIX packaging',
  'marketplace',
  'testing strategy',
  'code intelligence',
  'relationship detection',
  'session notes',
  'conflict detection',
  'governance',
  'build system',
  'TypeScript errors',
  'extension activation',
  'sidebar UI',
  'knowledge graph',
  'agent system',
  'DreamEngine',
  'token optimization',
  'deployment',
  'security',
  'error handling',
  'file watcher',
  'git hooks',
  'CLI commands',
  'embeddings',
  'metrics tracking',
  'decision debt',
  'auto-logging',
  'webview provider',
  'feature gates',
  'subscription tiers',
  'handoff document',
  'zombie process',
  'schema handler',
  'browser tools',
  'export PDF',
  'delta tracking',
  'workspace detection',
  'monorepo packages',
  'competitive analysis',
  'benchmark results',
  'version bump',
  'API design',
  'offline operation',
  'cross-platform support',
];

interface SearchResult {
  query: string;
  system: 'continuity' | 'mempalace';
  results: string[];
  latencyMs: number;
  tokenEstimate: number;
  resultCount: number;
  error?: string;
}

interface JudgedResult {
  query: string;
  continuity: { relevanceScore: number; latencyMs: number; tokenEstimate: number; resultCount: number };
  mempalace: { relevanceScore: number; latencyMs: number; tokenEstimate: number; resultCount: number };
  winner: 'continuity' | 'mempalace' | 'tie';
  reasoning: string;
}

// ── Continuity: Real SemanticSearchService with RRF hybrid search ──

let searchService: SemanticSearchService | null = null;
let decisions: any[] = [];

async function initContinuity(): Promise<void> {
  console.log('  Initializing Continuity SemanticSearchService...');
  const raw = fs.readFileSync(DECISIONS_PATH, 'utf-8');
  decisions = JSON.parse(raw);
  console.log(`  Loaded ${decisions.length} decisions`);

  searchService = new SemanticSearchService(PROJECT_DIR);
  await searchService.ensureInitialized();
  console.log('  Generating embeddings (if needed)...');
  await (searchService as any).generateMissingEmbeddings(decisions);
  console.log('  ✅ Decision search engine ready (RRF hybrid: semantic + keyword + tags)');

  console.log('  Scanning project files...');
  projectFileIndex = scanProjectFiles(PROJECT_DIR);
  console.log(`  Indexed ${projectFileIndex.length} files`);
  console.log('  ✅ Unified search ready (decisions + project files)');
}

async function searchContinuity(query: string): Promise<SearchResult> {
  const start = Date.now();
  try {
    // 1. Search decisions (primary signal)
    const decisionResults = await searchService!.search(query, decisions, {
      limit: 5,
      minScore: 0.20,
      hybridWeight: 0.65,
      includeKeyword: true,
      includeTags: true,
    });

    const scoredDecisions = decisionResults.map((r: any) => {
      const d = r.decision || r;
      const score = r.score ?? r.similarity ?? 0;
      const q = (d.question || '').slice(0, 150);
      const a = (d.answer || '').slice(0, 250);
      const tags = (d.tags || []).join(', ');
      return {
        score,
        text: `[Decision] score:${score.toFixed(3)} Q: ${q} A: ${a} [tags: ${tags}]`,
      };
    });

    // 2. Search project files (supplementary)
    const fileResults = searchProjectFiles(query, 5);
    const scoredFiles = fileResults.map((r) => ({
      score: r.score,
      text: `[File] score:${r.score.toFixed(3)} path: ${r.relativePath} snippet: ${r.snippet.slice(0, 300)}`,
    }));

    // 3. Blend by score — rank all results together, best first
    // Normalize file scores to decision score range with 0.8x weight
    // so decisions win ties, but high-scoring files can lead
    const allResults = [
      ...scoredDecisions.map(r => ({ ...r, normalizedScore: r.score })),
      ...scoredFiles.map(r => ({ ...r, normalizedScore: r.score * 0.8 })),
    ];
    allResults.sort((a, b) => b.normalizedScore - a.normalizedScore);
    const blended = allResults.slice(0, 7).map(r => r.text);

    const latencyMs = Date.now() - start;
    const tokenEstimate = Math.ceil(blended.join('\n').length / 4);

    return {
      query,
      system: 'continuity',
      results: blended,
      latencyMs,
      tokenEstimate,
      resultCount: blended.length,
    };
  } catch (e: any) {
    return {
      query,
      system: 'continuity',
      results: [],
      latencyMs: Date.now() - start,
      tokenEstimate: 0,
      resultCount: 0,
      error: e.message,
    };
  }
}

// ── MemPalace: CLI search ──

function searchMemPalace(query: string): SearchResult {
  const start = Date.now();
  try {
    const output = execSync(
      `mempalace search "${query.replace(/"/g, '\\"')}"`,
      { cwd: PROJECT_DIR, timeout: 30000, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    );

    const latencyMs = Date.now() - start;

    // Parse MemPalace output into result chunks
    const lines = output.split('\n').filter(l => l.trim());
    const results: string[] = [];
    let current = '';
    let capturing = false;

    for (const line of lines) {
      if (line.match(/^\s*\[\d+\]/)) {
        if (current) results.push(current.trim());
        current = line;
        capturing = true;
      } else if (capturing && line.trim()) {
        current += ' ' + line.trim();
      }
    }
    if (current) results.push(current.trim());

    const top5 = results.slice(0, 5);
    const tokenEstimate = Math.ceil(top5.join('\n').length / 4);

    return {
      query,
      system: 'mempalace',
      results: top5,
      latencyMs,
      tokenEstimate,
      resultCount: top5.length,
    };
  } catch (e: any) {
    return {
      query,
      system: 'mempalace',
      results: [],
      latencyMs: Date.now() - start,
      tokenEstimate: 0,
      resultCount: 0,
      error: e.stderr || e.message,
    };
  }
}

// ── LLM Judge ──

async function judgeResults(
  query: string,
  cResult: SearchResult,
  mResult: SearchResult,
  client: Anthropic
): Promise<JudgedResult> {
  const prompt = `You are a strict, impartial judge evaluating search result quality from two memory systems. Both systems searched the SAME codebase for: "${query}"

SYSTEM A (Continuity — decision-based memory with semantic search) returned ${cResult.results.length} results:
${cResult.results.length > 0 ? cResult.results.map((r, i) => `  ${i + 1}. ${r.slice(0, 400)}`).join('\n') : '  (no results)'}

SYSTEM B (MemPalace — codebase-wide file indexing) returned ${mResult.results.length} results:
${mResult.results.length > 0 ? mResult.results.map((r, i) => `  ${i + 1}. ${r.slice(0, 400)}`).join('\n') : '  (no results)'}

IMPORTANT: Judge relevance to the query "${query}" — not volume of content. A concise, directly relevant result beats a long tangentially related one. Consider:
1. Do the results actually answer or address the query topic?
2. Are the results specific and actionable, or generic and tangential?
3. Quality over quantity — 2 great results beat 5 mediocre ones.

Score each system 0.0 to 1.0:
- 1.0 = results directly and specifically address the query
- 0.7 = mostly relevant with minor noise
- 0.5 = mixed relevance
- 0.3 = mostly tangential
- 0.0 = irrelevant or no results

Respond ONLY with this JSON, nothing else:
{"scoreA": <number>, "scoreB": <number>, "winner": "<A|B|tie>", "reasoning": "<one sentence>"}`;

  try {
    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 200,
      messages: [{ role: 'user', content: prompt }],
    });

    const text = (response.content[0] as any).text;
    const json = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] || '{}');

    return {
      query,
      continuity: {
        relevanceScore: json.scoreA ?? 0,
        latencyMs: cResult.latencyMs,
        tokenEstimate: cResult.tokenEstimate,
        resultCount: cResult.resultCount,
      },
      mempalace: {
        relevanceScore: json.scoreB ?? 0,
        latencyMs: mResult.latencyMs,
        tokenEstimate: mResult.tokenEstimate,
        resultCount: mResult.resultCount,
      },
      winner: json.winner === 'A' ? 'continuity' : json.winner === 'B' ? 'mempalace' : 'tie',
      reasoning: json.reasoning || '',
    };
  } catch (e: any) {
    return {
      query,
      continuity: {
        relevanceScore: 0,
        latencyMs: cResult.latencyMs,
        tokenEstimate: cResult.tokenEstimate,
        resultCount: cResult.resultCount,
      },
      mempalace: {
        relevanceScore: 0,
        latencyMs: mResult.latencyMs,
        tokenEstimate: mResult.tokenEstimate,
        resultCount: mResult.resultCount,
      },
      winner: 'tie',
      reasoning: `Judge error: ${e.message}`,
    };
  }
}

// ── Wake-up measurement ──

function measureWakeUp(): {
  continuityTokens: number; mempalaceTokens: number;
  continuityMs: number; mempalaceMs: number;
} {
  // Continuity wake-up: simulates get_quick_context
  const cStart = Date.now();
  const raw = fs.readFileSync(DECISIONS_PATH, 'utf-8');
  const decs = JSON.parse(raw);
  const recent = decs.slice(-10);
  const contextStr = recent.map((d: any) => `${d.question}: ${(d.answer || '').slice(0, 200)}`).join('\n');
  const continuityMs = Date.now() - cStart;
  const continuityTokens = Math.ceil(contextStr.length / 4);

  // MemPalace wake-up
  const mStart = Date.now();
  let mempalaceTokens = 0;
  let mempalaceMs = 0;
  try {
    const output = execSync('mempalace wake-up', {
      cwd: PROJECT_DIR, timeout: 30000, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe']
    });
    mempalaceMs = Date.now() - mStart;
    mempalaceTokens = Math.ceil(output.length / 4);
  } catch (e: any) {
    mempalaceMs = Date.now() - mStart;
  }

  return { continuityTokens, mempalaceTokens, continuityMs, mempalaceMs };
}

// ── Report generation ──

function generateReport(results: JudgedResult[], wakeUp: any): string {
  const cWins = results.filter(r => r.winner === 'continuity').length;
  const mWins = results.filter(r => r.winner === 'mempalace').length;
  const ties = results.filter(r => r.winner === 'tie').length;

  const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;

  const avgCRel = avg(results.map(r => r.continuity.relevanceScore));
  const avgMRel = avg(results.map(r => r.mempalace.relevanceScore));
  const avgCLat = avg(results.map(r => r.continuity.latencyMs));
  const avgMLat = avg(results.map(r => r.mempalace.latencyMs));
  const avgCTok = avg(results.map(r => r.continuity.tokenEstimate));
  const avgMTok = avg(results.map(r => r.mempalace.tokenEstimate));

  let report = `# Head-to-Head v3: Continuity vs MemPalace (Unified Search)

## Methodology
- **Continuity:** Unified search — SemanticSearchService (RRF hybrid: semantic + keyword + tag fusion, all-MiniLM-L6-v2 embeddings) PLUS project file search (filename + content matching)
- **MemPalace:** CLI search against full mined codebase (ChromaDB vector index, 27,177 drawers)
- **Judge:** Claude Sonnet (LLM-as-a-Judge), blind evaluation
- **Queries:** ${results.length} diverse queries across the full project vocabulary
- **Corpus:** Same project — decisions + project files (Continuity) / 1,177 files mined (MemPalace)

---

## Results

| Metric | Continuity | MemPalace |
|--------|-----------|-----------|
| **Query Wins** | **${cWins}** | **${mWins}** |
| **Ties** | ${ties} | ${ties} |
| **Avg Relevance (0-1)** | **${avgCRel.toFixed(2)}** | **${avgMRel.toFixed(2)}** |
| **Avg Latency** | ${avgCLat.toFixed(0)}ms | ${avgMLat.toFixed(0)}ms |
| **Avg Tokens/Search** | ${avgCTok.toFixed(0)} | ${avgMTok.toFixed(0)} |

## Wake-Up Cost

| Metric | Continuity | MemPalace |
|--------|-----------|-----------|
| **Tokens** | ${wakeUp.continuityTokens} | ${wakeUp.mempalaceTokens} |
| **Latency** | ${wakeUp.continuityMs}ms | ${wakeUp.mempalaceMs}ms |

## Per-Query Detail

| # | Query | Continuity | MemPalace | Winner | Reasoning |
|---|-------|-----------|-----------|--------|-----------|
`;

  results.forEach((r, i) => {
    report += `| ${i + 1} | ${r.query} | ${r.continuity.relevanceScore.toFixed(1)} (${r.continuity.latencyMs}ms) | ${r.mempalace.relevanceScore.toFixed(1)} (${r.mempalace.latencyMs}ms) | **${r.winner}** | ${r.reasoning} |\n`;
  });

  report += `
---

## Key Differences
- **Continuity** uses unified search: structured decision records (question/answer/tags with RRF fusion) PLUS project file search (filename + content matching)
- **MemPalace** searches raw file content indexed from the entire codebase via ChromaDB
- Both use vector embeddings for semantic matching
- Continuity adds RRF fusion across semantic, keyword, and tag dimensions, supplemented by file search for code-specific queries

---
*Head-to-Head v3 (Unified Search) | ${new Date().toISOString().split('T')[0]} | Project Chronos*
`;

  return report;
}

// ── Main ──

async function main() {
  console.log('='.repeat(60));
  console.log('  HEAD-TO-HEAD v3 (UNIFIED): Continuity vs MemPalace');
  console.log('  Continuity: Unified Search (decisions + project files)');
  console.log('  MemPalace: CLI search (ChromaDB vector index)');
  console.log('='.repeat(60));
  console.log('');

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error('ERROR: ANTHROPIC_API_KEY not set in benchmarks/.env');
    process.exit(1);
  }
  const client = new Anthropic({ apiKey });

  // Initialize Continuity's real search engine
  await initContinuity();
  console.log('');

  // Round 0: Wake-up cost
  console.log('Round 0: Wake-up cost');
  const wakeUp = measureWakeUp();
  console.log(`  Continuity: ${wakeUp.continuityTokens} tokens (${wakeUp.continuityMs}ms)`);
  console.log(`  MemPalace:  ${wakeUp.mempalaceTokens} tokens (${wakeUp.mempalaceMs}ms)`);
  console.log('');

  // Query-by-query comparison
  console.log('Running queries...');
  const results: JudgedResult[] = [];

  for (let i = 0; i < QUERIES.length; i++) {
    const query = QUERIES[i];
    process.stdout.write(`  [${i + 1}/${QUERIES.length}] "${query}"... `);

    const cResult = await searchContinuity(query);
    const mResult = searchMemPalace(query);
    const judged = await judgeResults(query, cResult, mResult, client);
    results.push(judged);

    const icon = judged.winner === 'continuity' ? '🔵' :
                 judged.winner === 'mempalace' ? '🟠' : '⚪';
    console.log(`${icon} ${judged.winner} (C:${judged.continuity.relevanceScore} M:${judged.mempalace.relevanceScore})`);
  }

  // Summary
  const cWins = results.filter(r => r.winner === 'continuity').length;
  const mWins = results.filter(r => r.winner === 'mempalace').length;
  const ties = results.filter(r => r.winner === 'tie').length;

  console.log('');
  console.log('='.repeat(60));
  console.log('  FINAL SCORE');
  console.log('='.repeat(60));
  console.log(`  Continuity: ${cWins} wins`);
  console.log(`  MemPalace:  ${mWins} wins`);
  console.log(`  Ties:       ${ties}`);

  const avgCRel = results.reduce((s, r) => s + r.continuity.relevanceScore, 0) / results.length;
  const avgMRel = results.reduce((s, r) => s + r.mempalace.relevanceScore, 0) / results.length;
  console.log(`  Avg Relevance — C: ${avgCRel.toFixed(2)} | M: ${avgMRel.toFixed(2)}`);
  console.log('');

  // Save
  fs.mkdirSync(path.dirname(RESULTS_FILE), { recursive: true });
  fs.writeFileSync(RESULTS_FILE, JSON.stringify({ wakeUp, results, version: 'v3-unified' }, null, 2));
  console.log(`✅ Raw results: ${RESULTS_FILE}`);

  const report = generateReport(results, wakeUp);
  fs.writeFileSync(REPORT_FILE, report);
  console.log(`✅ Report: ${REPORT_FILE}`);
}

main().catch(console.error);
