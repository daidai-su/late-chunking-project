# Boundary-Aware Chunking Results Template

Do not fill in results until the optional boundary-aware chunking notebook has
actually executed and saved artifacts.

## Research Question

Does aligning chunk spans to sentence or paragraph boundaries improve
traditional chunking, Late Chunking, or both, compared with fixed 256-token
spans?

## Setup

- Date:
- Colab runtime:
- GPU:
- Project commit:
- Official late-chunking commit:
- Task(s): `SciFactChunked` by default
- Model:
- Max chunk tokens: `256`
- Overlap tokens: `64`
- Chunking methods:
  - `fixed_256_tokens`
  - `sentence_boundary_approx`
  - `paragraph_boundary_approx`
  - `overlap_fixed`

## Main Results

| Task | Chunking method | Pooling | nDCG@10 | Recall@100 | MRR@10 | Chunks | Avg chunk length | Wall time | Memory estimate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFactChunked | fixed_256_tokens | traditional_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | fixed_256_tokens | late_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | sentence_boundary_approx | traditional_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | sentence_boundary_approx | late_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | paragraph_boundary_approx | traditional_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | paragraph_boundary_approx | late_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | overlap_fixed | traditional_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SciFactChunked | overlap_fixed | late_chunking | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Required Comparisons

- Does `sentence_boundary_approx` improve traditional chunking?
- Does `sentence_boundary_approx` improve Late Chunking?
- Does `paragraph_boundary_approx` improve traditional chunking?
- Does `paragraph_boundary_approx` improve Late Chunking?
- Does boundary-aware chunking reduce the advantage of Late Chunking over
  traditional chunking?
- Does `overlap_fixed` improve enough to justify increased chunk count, runtime,
  and memory?
- Does any improvement concentrate on longer documents?

## Diagnostics

- Number of chunks by method:
- Average chunk length by method:
- Wall-clock run time by method:
- Indexing time:
- Retrieval time:
- Memory estimate:
- Long-document subset behavior:

Note: if the official MTEB path does not expose indexing and retrieval timing
separately, report combined wall-clock time and label indexing/retrieval timing
as unavailable rather than guessing.

## Limitations

- No model training was performed.
- No paid APIs or LLMs were used.
- Boundary splitting is approximate and punctuation/newline based.
- Do not tune chunk size on test; fixed settings are `max_chunk_tokens = 256`
  and `overlap_tokens = 64`.
- Report honestly if boundary-aware chunking does not improve.
