/**
 * Fixture + quiz loading for the benchmark runners.
 *
 * Fixtures live under fixtures/<project>/.continuity/decisions.json with a
 * matching prompts/quizzes/<project>.json. Schema documented in
 * docs/methodology.md.
 */

import * as fs from 'fs';
import * as path from 'path';

const FIXTURES_ROOT = path.resolve(__dirname, '..', '..', 'fixtures');
const QUIZZES_ROOT = path.resolve(__dirname, '..', '..', 'prompts/quizzes');

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
}

/**
 * Loads a fixture project by name from fixtures/<projectName>/.continuity/decisions.json.
 */
export function loadFixture(projectName: string): FixtureProject {
  const projectPath = path.join(FIXTURES_ROOT, projectName);
  const decisionsPath = path.join(projectPath, '.continuity', 'decisions.json');

  if (fs.existsSync(decisionsPath)) {
    return {
      name: projectName,
      rootPath: projectPath,
      decisions: readDecisions(decisionsPath),
    };
  }

  throw new Error(`Fixture not found: ${projectName}. Expected at ${decisionsPath}.`);
}

/**
 * Loads a quiz JSON by project name from prompts/quizzes/<projectName>.json.
 * If no quiz file exists, auto-generates a 20-question direct-recall quiz from
 * the fixture's decisions.
 */
export function loadQuiz(projectName: string, fallbackFixture?: FixtureProject): Quiz {
  const quizPath = path.join(QUIZZES_ROOT, `${projectName}.json`);
  if (fs.existsSync(quizPath)) {
    const raw = fs.readFileSync(quizPath, 'utf-8');
    const parsed = JSON.parse(raw) as Quiz;
    if (!Array.isArray(parsed.questions)) {
      throw new Error(`Quiz at ${quizPath} is malformed (missing questions array)`);
    }
    return parsed;
  }

  if (!fallbackFixture) {
    throw new Error(`Quiz not found: ${quizPath} and no fixture fallback provided.`);
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

export const REPORTS_DIR = path.resolve(__dirname, '..', '..', 'reports');

export function ensureReportsDir(): void {
  fs.mkdirSync(REPORTS_DIR, { recursive: true });
}
