# Phase A Colab Notes

These notes summarize the provisional Colab run observed during Phase A. Keep
the raw artifacts and rerun cleanly before treating these as final report
numbers.

## Observed Run

- Date: 2026-06-27
- Runtime: Google Colab GPU
- GPU: Tesla T4
- Project commit used in Colab: `135e0b0d77efc39c61fdb9c14c0de7658a2b8429`
- Official repository commit: `1d3bb02bf091becd0771455e4e7959463935e26c`
- Mode: `small`
- Task: `SciFactChunked`
- Official command: `python run_chunked_eval.py --task-name SciFactChunked`
- Command return code: `0`

## Important Discovery

The official command completed successfully, but the final metrics were not
printed in stdout. MTEB wrote result JSON files under the official repository:

- `/content/official_late_chunking/results-normal-pooling/no_model_name_available/no_revision_available/SciFactChunked.json`
- `/content/official_late_chunking/results-chunked-pooling/no_model_name_available/no_revision_available/SciFactChunked.json`

The notebook should parse those JSON files after the official command finishes.

## Observed Metrics

| Task | Method | nDCG@10 | Recall@10 | MAP@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFactChunked | normal_pooling | 0.64198 | 0.76589 | 0.59670 | 0.61138 |
| SciFactChunked | chunked_pooling | 0.66098 | 0.77756 | 0.61829 | 0.63372 |

## Colab Dependency Notes

The Colab runtime had optional media packages that interfered with text-only
imports:

- `torchcodec`
- `torchvision`

The notebook removes these optional packages because this project does not use
audio, video, or image decoding.

## Artifacts To Save

- `/content/late_chunking_outputs/run_manifest.json`
- `/content/late_chunking_outputs/tables/official_results_summary.csv`
- `/content/late_chunking_outputs/logs/official_SciFactChunked_stdout.txt`
- `/content/late_chunking_outputs/logs/official_SciFactChunked_stderr.txt`

## Final Clean Run Checklist

1. Start a fresh Colab GPU runtime.
2. Run the notebook from the top.
3. Use `MODE = "small"` and `TASKS = ["SciFactChunked"]`.
4. Confirm `metrics_summary.csv` has `returncode = 0`.
5. Confirm `official_results_summary.csv` contains `normal_pooling` and
   `chunked_pooling`.
6. Confirm `run_manifest.json` has `metrics_parsed_successfully = true`.

