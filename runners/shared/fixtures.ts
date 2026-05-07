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

// Two supported repo layouts:
//
//   (a) continuity-ultimate (this repo): runners/shared/ at
//         verification/shared/id-rag-parallel/runners/shared/, so 5
//         levels up to repo root. Fixtures + quizzes live under
//         verification/shared/id-rag-parallel/.
//
//   (b) continuity-benchmarks (public repo): runners/shared/ at the top
//         level, so 2 levels up to repo root. Fixtures at fixtures/,
//         quizzes at prompts/quizzes/.
//
// The loader checks both layouts in order and uses whichever exists, so
// the same runner source works in both repos with no path rewriting at
// staging time.
const CANDIDATE_LAYOUTS: Array<{ fixtures: string; quizzes: string; legacyPaydash?: string }> = [
  // (a) continuity-ultimate layout
  {
    fixtures: path.resolve(__dirname, '..', '..', '..', '..', '..', 'verification/shared/id-rag-parallel/fixtures'),
    quizzes: path.resolve(__dirname, '..', '..', '..', '..', '..', 'verification/shared/id-rag-parallel/quizzes'),
    legacyPaydash: path.resolve(__dirname, '..', '..', '..', '..', '..', 'demo-projects/peer-review/with-continuity/.continuity/decisions.json'),
  },
  // (b) continuity-benchmarks public repo layout
  {
    fixtures: path.resolve(__dirname, '..', '..', 'fixtures'),
    quizzes: path.resolve(__dirname, '..', '..', 'prompts/quizzes'),
  },
];

function findFixturesRoot(projectName: string): { fixturesRoot: string; legacyPaydash?: string } {
  for (const layout of CANDIDATE_LAYOUTS) {
    const candidate = path.join(layout.fixtures, projectName, '.continuity', 'decisions.json');
    if (fs.existsSync(candidate)) return { fixturesRoot: layout.fixtures, legacyPaydash: layout.legacyPaydash };
    if (projectName === 'paydash-api' && layout.legacyPaydash && fs.existsSync(layout.legacyPaydash)) {
      return { fixturesRoot: layout.fixtures, legacyPaydash: layout.legacyPaydash };
    }
  }
  return { fixturesRoot: CANDIDATE_LAYOUTS[0].fixtures, legacyPaydash: CANDIDATE_LAYOUTS[0].legacyPaydash };
}

function findQuizzesRoot(projectName: string): string {
  for (const layout of CANDIDATE_LAYOUTS) {
    if (fs.existsSync(path.join(layout.quizzes, `${projectName}.json`))) return layout.quizzes;
  }
  return CANDIDATE_LAYOUTS[0].quizzes;
}

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
  const { fixturesRoot, legacyPaydash } = findFixturesRoot(projectName);
  const clioPath = path.join(fixturesRoot, projectName);
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

  if (projectName === 'paydash-api' && legacyPaydash && fs.existsSync(legacyPaydash)) {
    const decisions = readDecisions(legacyPaydash);
    return {
      name: projectName,
      rootPath: path.dirname(path.dirname(legacyPaydash)),
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
  const clioQuizPath = path.join(findQuizzesRoot(projectName), `${projectName}.json`);
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

// Reports dir: continuity-ultimate uses benchmarks/reports/ (5 levels up
// from runners/shared/), continuity-benchmarks uses reports/ (2 levels up).
function findReportsDir(): string {
  const candidates = [
    path.resolve(__dirname, '..', '..', '..', '..', '..', 'benchmarks/reports'),
    path.resolve(__dirname, '..', '..', 'reports'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  // Default to first candidate if neither exists yet (will be created by ensureReportsDir)
  return candidates[0];
}

export const REPORTS_DIR = findReportsDir();

export function ensureReportsDir(): void {
  fs.mkdirSync(REPORTS_DIR, { recursive: true });
}
