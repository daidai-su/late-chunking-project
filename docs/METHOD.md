# Method

This project started as a Phase A scaffold for reproducing the official Late
Chunking baseline on small BEIR-style chunked retrieval tasks. Phase B adds
training-free chunk-to-document aggregation experiments while preserving the
official baseline path.

## Baselines

Traditional chunking splits a document into chunks first, then independently
encodes each chunk. Late chunking encodes the full document or long context
first, then mean-pools token representations by chunk span. The Phase A
notebook uses the official `jina-ai/late-chunking` repository as the primary
implementation source and launches:

```bash
python run_chunked_eval.py --task-name SciFactChunked
```

from the official repository directory.

## Scope

Phase A does not add new research methods, train models, or report unverified
results. The local package only provides environment inspection, run logging,
metric parsing, and small CPU-only utilities that keep later experiments tidy.

## Future Hooks

Later aggregation experiments should be added behind separate modules or
notebook sections so the official baseline path remains auditable.

## Phase B Aggregation

The research extension keeps chunk-level late chunk embeddings fixed and
changes only how retrieved chunks are aggregated into document rankings. It does
not fine-tune embedding models, train neural models, use paid APIs, or call
LLMs.

The official evaluator creates chunk IDs in the form `{doc_id}~{chunk_index}`
and converts chunk rankings to document rankings by retaining the maximum chunk
score per document. Because the chunk ranking is sorted by descending score,
this is equivalent to a first occurrence rule in ordinary non-tied cases.

Phase B evaluates:

- `first_occurrence`
- `max_score`
- `topk_mean`
- `softmax_topk`
- `max_plus_coverage`
- `rrf_chunk_vote`
- `bm25_only`
- `late_first_occurrence_plus_bm25_rrf`
- `late_softmax_topk_plus_bm25_rrf`

BM25 fusion is reported separately as a hybrid dense + lexical retrieval
baseline, not as a pure Late Chunking improvement.

See `docs/OFFICIAL_CODE_AUDIT.md` for the official source audit and
`notebooks/02_late_chunking_aggregation_experiment_colab.ipynb` for the
Colab experiment workflow.

