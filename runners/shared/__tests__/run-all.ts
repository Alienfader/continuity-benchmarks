/**
 * Runs all node:test files in this directory.
 *
 * Invoke with:
 *   npx ts-node benchmarks/src/id-rag-parallel/runners/shared/__tests__/run-all.ts
 *
 * We don't hook into the repo's jest suite because jest.config.js's `roots`
 * don't include benchmarks/, and the Atlas scope forbids editing jest config.
 * node:test is zero-dependency and ships with Node >= 18, so these tests are
 * portable and CI-friendly.
 */

import { run } from 'node:test';
import * as path from 'path';
import * as fs from 'fs';

async function main() {
  const here = __dirname;
  const files = fs
    .readdirSync(here)
    .filter((f) => f.endsWith('.test.ts'))
    .map((f) => path.join(here, f));

  if (files.length === 0) {
    console.error('No .test.ts files found in', here);
    process.exit(1);
  }

  let passed = 0;
  let failed = 0;
  const stream = run({ files });

  stream.on('test:pass', (evt) => {
    if (evt && (evt as { todo?: unknown }).todo) return;
    // Only count leaf tests (ignore suites)
    const nesting = (evt as { nesting?: number }).nesting ?? 0;
    if (nesting === 0) {
      passed += 1;
      console.log(`  ✓ ${(evt as { name: string }).name}`);
    }
  });
  stream.on('test:fail', (evt) => {
    const nesting = (evt as { nesting?: number }).nesting ?? 0;
    if (nesting === 0) {
      failed += 1;
      const name = (evt as { name: string }).name;
      const details = (evt as { details?: { error?: Error } }).details;
      console.log(`  ✗ ${name}`);
      if (details?.error) console.log(`    ${details.error.message}`);
    }
  });
  stream.on('end', () => {
    console.log(`\n${passed} passed, ${failed} failed`);
    if (failed > 0) process.exit(1);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
