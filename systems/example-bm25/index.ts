/**
 * example-bm25 — reference adapter showing the smallest possible
 * RetrievalSystem implementation. Wraps the built-in BM25Retriever
 * verbatim; running it should produce numbers identical to the
 * `continuity-in-loop` condition with prompt-keyed (not entity-keyed)
 * retrieval. Useful as a smoke test that the adapter plumbing works
 * end-to-end and as a copy-paste starting point.
 */

import type { RetrievalSystem } from '../../runners/shared/system-adapter';
import type { Decision } from '../../runners/shared/fixtures';
import type { Retriever } from '../../runners/shared/retrieval';
import { BM25Retriever } from '../../runners/shared/retrieval';

const exampleBM25: RetrievalSystem = {
  name: 'example-bm25',
  description: 'Reference adapter — wraps the built-in BM25Retriever verbatim',

  init(decisions: Decision[]): Retriever {
    return new BM25Retriever(decisions);
  },
};

export default exampleBM25;
