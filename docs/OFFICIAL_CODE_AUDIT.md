# Official Code Audit

This audit was prepared before implementing Phase B aggregation wrappers. It
inspects the official `jina-ai/late-chunking` code without modifying official
files.

Inspected source:

- Official repository: <https://github.com/jina-ai/late-chunking>
- Local audit clone: `.official_audit_tmp`
- Observed Colab commit during Phase A:
  `1d3bb02bf091becd0771455e4e7959463935e26c`

## Entry Point

`run_chunked_eval.py` is the official baseline entry point. The Phase A command
was:

```bash
python run_chunked_eval.py --task-name SciFactChunked
```

The script loads a task class from `chunked_pooling.chunked_eval_tasks`, loads
the embedding model through `chunked_pooling.wrappers.load_model`, then runs two
MTEB evaluations:

- `chunked_pooling_enabled=True`, saved under `results-chunked-pooling`
- `chunked_pooling_enabled=False`, saved under `results-normal-pooling`

## Where Corpus Is Chunked

Chunking is handled in `chunked_pooling/mteb_chunked_eval.py` and
`chunked_pooling/chunking.py`.

For non-late chunking, `_evaluate_monolingual()` calls `_apply_chunking()`,
which uses `Chunker.chunk()` to create token or sentence chunk spans and then
decodes chunk text. `_flatten_chunks()` converts those per-document chunk lists
into chunk documents for retrieval.

For late chunking, `_evaluate_monolingual()` computes chunk annotations via
`_calculate_annotations()` and applies `chunked_pooling()` to token
representations after full-document encoding.

## Where Embeddings Are Produced

For queries, `_evaluate_monolingual()` calls `model.encode_queries(query_texts)`
when available, otherwise `model.encode(query_texts)`.

For late-chunked corpus documents, `_evaluate_monolingual()` tokenizes each full
document, forwards it through the embedding model, then calls
`chunked_pooling()` with chunk annotations to produce one embedding per chunk.

For normal chunking, MTEB's `RetrievalEvaluator` encodes the already flattened
chunk corpus.

## Where Chunk-Level kNN Search Is Performed

For late chunking, `flatten_corpus_embs()` creates:

- `chunk_id_list`
- `doc_to_chunk`
- `flattened_corpus_embs`

Chunk IDs use the format:

```text
{doc_id}~{chunk_index}
```

The evaluator computes:

```python
similarity_matrix = np.dot(query_embs, flattened_corpus_embs.T)
```

Then `get_results()` converts each query row into a sorted chunk score mapping
and keeps the top `max(k_values)` chunks.

## Where Chunk Rankings Become Document Rankings

`AbsTaskChunkedRetrieval.get_doc_results()` converts chunk rankings to document
rankings. It extracts the document ID with:

```python
d_id = "~".join(c_id.split("~")[:-1])
```

and keeps the maximum chunk score seen for each document:

```python
if (d_id not in docs) or (score > docs[d_id]):
    docs[d_id] = float(score)
```

Because `get_results()` already sorts chunks by descending score, this is
equivalent to retaining the first occurrence for normal non-tied rankings.
Phase B implements `first_occurrence` to match this behavior as closely as
possible, and adds alternative aggregation methods as wrappers around saved
chunk rankings.

## Where BEIR Metrics Are Computed

After document-level rankings are produced, `_evaluate_monolingual()` calls:

```python
RetrievalEvaluator.evaluate(...)
RetrievalEvaluator.evaluate_custom(..., "mrr")
```

The official JSON result files contain metrics such as:

- `ndcg_at_10`
- `map_at_10`
- `recall_at_10`
- `mrr_at_10`

The Phase B extension uses local metric helpers only for the new aggregation
rankings and validates those helpers with CPU-only toy tests.

## Phase B Adapter Boundary

The Phase B notebook does not edit official repository files. It temporarily
wraps `AbsTaskChunkedRetrieval.get_results()` inside the notebook process to
save chunk-level rankings as JSONL, then restores the original method. All new
chunk-to-document aggregation happens in `src/latechunk_project/`.

