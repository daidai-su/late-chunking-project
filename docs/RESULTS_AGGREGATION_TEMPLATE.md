# Aggregation Results Template

Do not fill in results until the aggregation notebook has actually executed.

## Research Question

Can query-aware chunk-to-document aggregation improve document-level retrieval
from chunk-level late chunk embeddings without training?

## Setup

- Date:
- Colab runtime:
- GPU:
- Project commit:
- Official late-chunking commit:
- Task(s):
- Chunk top K for aggregation:
- BM25 enabled:
- RRF fusion enabled:

## Main Results

| Method family | Method | nDCG@10 | Recall@100 | MRR@10 | Latency/query |
| --- | --- | ---: | ---: | ---: | ---: |
| official_baseline | first_occurrence | TBD | TBD | TBD | TBD |
| late_chunk_aggregation | max_score | TBD | TBD | TBD | TBD |
| late_chunk_aggregation | topk_mean | TBD | TBD | TBD | TBD |
| late_chunk_aggregation | softmax_topk | TBD | TBD | TBD | TBD |
| late_chunk_aggregation | max_plus_coverage | TBD | TBD | TBD | TBD |
| late_chunk_aggregation | rrf_chunk_vote | TBD | TBD | TBD | TBD |
| lexical | bm25_only | TBD | TBD | TBD | TBD |
| hybrid | late_first_occurrence_plus_bm25_rrf | TBD | TBD | TBD | TBD |
| hybrid | late_softmax_topk_plus_bm25_rrf | TBD | TBD | TBD | TBD |

## Per-Query Analysis

- Improved examples:
- Degraded examples:
- Relevant documents with repeated retrieved chunks:

## Diagnostics

- Number of chunks:
- Average chunks per document:
- Duplicate chunk fraction:
- Correlation between chunks per relevant doc and improvement:

## Limitations

- No model training was performed.
- Aggregation methods modify only chunk-to-document scoring.
- BM25 fusion is a hybrid retrieval baseline, not a pure Late Chunking method.
- Results are exploratory unless reproduced across multiple tasks.

