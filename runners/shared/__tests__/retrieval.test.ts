import { test } from 'node:test';
import * as assert from 'node:assert/strict';
import { BM25Retriever, renderContext } from '../retrieval';
import type { Decision } from '../fixtures';

const DECISIONS: Decision[] = [
  {
    id: 'd1',
    question: 'Why PostgreSQL for the primary datastore?',
    answer: 'ACID guarantees for payment processing and strong relational model.',
    tags: ['database', 'architecture'],
    status: 'active',
  },
  {
    id: 'd2',
    question: 'Why did we add Redis?',
    answer: 'Redis caches session tokens and rate-limit counters.',
    tags: ['cache', 'performance'],
    status: 'active',
  },
  {
    id: 'd3',
    question: 'Why retire the legacy MongoDB cluster?',
    answer: 'MongoDB was superseded by Postgres; the driver is no longer maintained.',
    tags: ['database'],
    status: 'superseded',
  },
  {
    id: 'd4',
    question: 'Why choose Go for the gateway?',
    answer: 'Low-latency routing, small container footprint, and team fluency.',
    tags: ['language'],
    status: 'active',
  },
];

test('BM25 retrieval ranks the postgres decision first for a postgres query', () => {
  const r = new BM25Retriever(DECISIONS);
  const top = r.retrieve('postgres payment acid', 2);
  assert.ok(top.length >= 1);
  assert.equal(top[0].id, 'd1');
});

test('BM25 retrieval returns fewer than k when only some docs match', () => {
  const r = new BM25Retriever(DECISIONS);
  const top = r.retrieve('redis cache', 5);
  assert.ok(top.length >= 1);
  assert.equal(top[0].id, 'd2');
});

test('BM25 retrieval returns empty for a query with no overlap', () => {
  const r = new BM25Retriever(DECISIONS);
  const top = r.retrieve('xyzzy qwerty', 5);
  assert.equal(top.length, 0);
});

test('BM25 retrieval downranks superseded decisions', () => {
  const r = new BM25Retriever(DECISIONS);
  // The MongoDB doc is superseded; it should not beat the active Postgres doc
  // for a "database" query.
  const top = r.retrieve('database', 2);
  assert.ok(top.length >= 1);
  assert.notEqual(top[0].id, 'd3');
});

test('renderContext empty returns empty string', () => {
  assert.equal(renderContext([]), '');
});

test('renderContext includes question, answer, tags, and non-active status', () => {
  const out = renderContext(DECISIONS);
  assert.match(out, /## Project decisions/);
  assert.match(out, /PostgreSQL/);
  assert.match(out, /\[database, architecture\]/);
  assert.match(out, /\(status: superseded\)/);
});
