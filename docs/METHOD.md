# Method

This project is a Phase A scaffold for reproducing the official Late Chunking
baseline on small BEIR-style chunked retrieval tasks.

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

