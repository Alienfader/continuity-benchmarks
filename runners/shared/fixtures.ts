/**
 * Fixture + quiz loading for the ID-RAG-parallel runners.
 *
 * Clio authors the fixture projects and quiz files. This module reads them
 * at runtime. If a fixture path is missing (e.g., the runner is executed
 * before Clio's branch merges), we fall back to the existing paydash-api
 * fixture at demo-projects/peer-review/with-continuity/, which has been the
 * canonical fixture since well before this benchmark suite.
 *
 * Schema matches the spec in PRIMING/clio.md.
 */

import * as fs from 'fs';
import * as path from 'path';

// Public-repo layout: runners/shared/ is 2 levels down from repo root.
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const FIXTURES_ROOT = path.resolve(REPO_ROOT, 'fixtures');
const QUIZZES_ROOT = path.resolve(REPO_ROOT, 'prompts', 'quizzes');
// paydash-api lives at fixtures/paydash-api/ in this repo; the "legacy" branch
// is no longer needed but kept as a no-op fallback.
const LEGACY_PAYDASH_DECISIONS = path.resolve(
  REPO_ROOT,
  'fixtures/paydash-api/.continuity/decisions.json',
);

export interface Decision {
  id: string;
  question: string;
  answer: string;
  tags?: string[];
  files?: string[];
  timestamp?: string;
  status?: 'active' | 'outdated' | 'superseded' | 'draft' | 'deprecated';
  relationships?: { supersedes?: string[]; relatedTo?: string[] };
  [key: string]: unknown;
}

export type QuestionType =
  | 'direct-recall'
  | 'supersedes-aware'
  | 'conflict-resolution'
  | 'application';

export interface QuizQuestion {
  id: string;
  question: string;
  groundTruth: string;
  type: QuestionType;
  /** For supersedes-aware questions: at which session index the superseding event fires. */
  correctAnswerAt?: string | number | null;
}

export interface Quiz {
  project: string;
  questions: QuizQuestion[];
}

export interface FixtureProject {
  name: string;
  rootPath: string;
  decisions: Decision[];
  source: 'clio' | 'legacy-paydash';
}

/**
 * Loads a fixture project by name. Checks Clio's location first; falls back
 * to the legacy paydash-api fixture for the name 'paydash-api'.
 */
export function loadFixture(projectName: string): FixtureProject {
  const clioPath = path.join(FIXTURES_ROOT, projectName);
  const clioDecisionsPath = path.join(clioPath, '.continuity', 'decisions.json');

  if (fs.existsSync(clioDecisionsPath)) {
    const decisions = readDecisions(clioDecisionsPath);
    return {
      name: projectName,
      rootPath: clioPath,
      decisions,
      source: 'clio',
    };
  }

  if (projectName === 'paydash-api' && fs.existsSync(LEGACY_PAYDASH_DECISIONS)) {
    const decisions = readDecisions(LEGACY_PAYDASH_DECISIONS);
    return {
      name: projectName,
      rootPath: path.dirname(path.dirname(LEGACY_PAYDASH_DECISIONS)),
      decisions,
      source: 'legacy-paydash',
    };
  }

  throw new Error(
    `Fixture not found: ${projectName}. Expected at ${clioDecisionsPath} or the legacy paydash path.`,
  );
}

/**
 * Loads a quiz JSON by project name. Falls back to a minimal auto-generated
 * quiz from the fixture decisions if no quiz file exists (for dry-runs before
 * Clio's branch merges).
 */
export function loadQuiz(projectName: string, fallbackFixture?: FixtureProject): Quiz {
  const clioQuizPath = path.join(QUIZZES_ROOT, `${projectName}.json`);
  if (fs.existsSync(clioQuizPath)) {
    const raw = fs.readFileSync(clioQuizPath, 'utf-8');
    const parsed = JSON.parse(raw) as Quiz;
    if (!Array.isArray(parsed.questions)) {
      throw new Error(`Quiz at ${clioQuizPath} is malformed (missing questions array)`);
    }
    return parsed;
  }

  if (!fallbackFixture) {
    throw new Error(`Quiz not found: ${clioQuizPath} and no fixture fallback provided.`);
  }

  // Auto-generate a 20-question direct-recall quiz from fixture decisions.
  const questions: QuizQuestion[] = fallbackFixture.decisions
    .slice(0, 20)
    .map((d, i) => ({
      id: `Q${i + 1}`,
      question: d.question,
      groundTruth: d.answer,
      type: 'direct-recall' as const,
      correctAnswerAt: null,
    }));

  // Pad to 20 if fewer decisions by repeating (keeps scoring shape consistent).
  while (questions.length < 20 && fallbackFixture.decisions.length > 0) {
    const d = fallbackFixture.decisions[questions.length % fallbackFixture.decisions.length];
    questions.push({
      id: `Q${questions.length + 1}`,
      question: d.question,
      groundTruth: d.answer,
      type: 'direct-recall',
      correctAnswerAt: null,
    });
  }

  return { project: projectName, questions };
}

function readDecisions(p: string): Decision[] {
  const raw = fs.readFileSync(p, 'utf-8');
  const parsed = JSON.parse(raw) as Decision[] | { decisions: Decision[] };
  if (Array.isArray(parsed)) return parsed;
  if (parsed && Array.isArray(parsed.decisions)) return parsed.decisions;
  throw new Error(`decisions.json at ${p} is neither an array nor { decisions: [...] }`);
}

export const REPORTS_DIR = path.resolve(REPO_ROOT, 'reports');

export function ensureReportsDir(): void {
  fs.mkdirSync(REPORTS_DIR, { recursive: true });
}
