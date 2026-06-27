"""Generate the Phase B aggregation experiment Colab notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_late_chunking_aggregation_experiment_colab.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "colab": {"name": "02_late_chunking_aggregation_experiment_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        markdown(
            """
            # Late Chunking Aggregation Experiment

            Phase B evaluates training-free chunk-to-document aggregation over
            chunk-level Late Chunking rankings. It keeps the official baseline
            auditable and writes all new artifacts under a separate output
            subdirectory.
            """
        ),
        code(
            '''
            PROJECT_NAME = "late_chunking_aggregation"
            PUBLIC_REPO_URL = "https://github.com/<YOUR_GITHUB_USERNAME>/late-chunking-project.git"
            OFFICIAL_REPO_URL = "https://github.com/jina-ai/late-chunking.git"

            MODE = "small"  # smoke / small / full
            TASKS = ["SciFactChunked"]
            OPTIONAL_TASKS = ["NFCorpusChunked"]

            SEED = 42
            USE_GPU_IF_AVAILABLE = True
            FORCE_RERUN = False
            CONFIRM_FULL_RUN = False

            OUTPUT_DIR = "/content/late_chunking_outputs"
            AGG_OUTPUT_SUBDIR = "aggregation_experiment"
            OFFICIAL_REPO_DIR = "/content/official_late_chunking"

            CHUNK_TOP_K_FOR_AGGREGATION = 1000
            DOC_TOP_K_EVAL = [10, 100]

            TOPK_MEAN_K = 3
            SOFTMAX_TOPK_K = 3
            SOFTMAX_TAU = 0.05
            RRF_K = 60

            RUN_BM25 = True
            RUN_RRF_FUSION = True
            '''
        ),
        markdown("## Install Project Dependencies"),
        code(
            r'''
            import os
            import subprocess
            import sys
            from pathlib import Path

            def run_command(command, cwd=None, check=True):
                print("+", " ".join(str(part) for part in command))
                return subprocess.run(command, cwd=cwd, check=check)

            def find_or_clone_project():
                current = Path.cwd()
                candidates = [
                    current,
                    current.parent if current.name == "notebooks" else current,
                    Path("/content") / "late_chunking_project",
                    Path("/content") / PROJECT_NAME,
                ]
                for candidate in candidates:
                    if (candidate / "src" / "latechunk_project").exists():
                        return candidate

                target = Path("/content") / "late_chunking_project"
                if "<YOUR_GITHUB_USERNAME>" in PUBLIC_REPO_URL:
                    raise RuntimeError(
                        "Edit PUBLIC_REPO_URL before running this notebook standalone in Colab, "
                        "or clone the project repository before running."
                    )
                if not target.exists():
                    run_command(["git", "clone", PUBLIC_REPO_URL, str(target)])
                return target

            PROJECT_ROOT = find_or_clone_project()
            os.chdir(PROJECT_ROOT)
            print(f"Project root: {PROJECT_ROOT}")

            src_path = str(PROJECT_ROOT / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            requirements_path = PROJECT_ROOT / "requirements-colab.txt"
            if requirements_path.exists():
                run_command([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)])

            if (PROJECT_ROOT / "pyproject.toml").exists() or (PROJECT_ROOT / "setup.py").exists():
                run_command([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)], check=False)

            from latechunk_project.dependency_workarounds import (
                ensure_numpy_stack_healthy,
                remove_optional_media_dependencies,
            )

            numpy_stack_status = ensure_numpy_stack_healthy()
            print("NumPy stack check:", numpy_stack_status)
            dependency_workaround_events = remove_optional_media_dependencies()
            print("Optional media dependency checks:", dependency_workaround_events)
            '''
        ),
        markdown("## Clone Official Repository"),
        code(
            '''
            import sys
            from pathlib import Path

            from latechunk_project.dependency_workarounds import (
                ensure_numpy_stack_healthy,
                remove_optional_media_dependencies,
            )
            from latechunk_project.official_repo import clone_official_repo, install_official_repo
            from latechunk_project.run_utils import ensure_output_tree

            output_dirs = ensure_output_tree(OUTPUT_DIR)
            AGG_OUTPUT_DIR = Path(OUTPUT_DIR) / AGG_OUTPUT_SUBDIR
            for name in ["rankings", "tables", "figures", "logs"]:
                (AGG_OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

            official_repo_path = clone_official_repo(
                OFFICIAL_REPO_URL,
                OFFICIAL_REPO_DIR,
                force_rerun=FORCE_RERUN,
            )
            official_install = install_official_repo(official_repo_path)
            numpy_stack_status_after_official_install = ensure_numpy_stack_healthy()
            dependency_workaround_events.extend(remove_optional_media_dependencies())

            official_src = str(official_repo_path)
            if official_src not in sys.path:
                sys.path.insert(0, official_src)

            print(f"Official repo: {official_repo_path}")
            print(f"Official editable install return code: {official_install.returncode}")
            print("NumPy stack check after official install:", numpy_stack_status_after_official_install)
            print(f"Aggregation outputs: {AGG_OUTPUT_DIR}")
            '''
        ),
        markdown("## Environment And Run Manifest"),
        code(
            '''
            import json
            from pathlib import Path

            from latechunk_project.env_utils import collect_environment_info, print_environment_info
            from latechunk_project.run_utils import utc_now_iso, write_json

            if MODE == "full" and not CONFIRM_FULL_RUN:
                raise SystemExit("Set CONFIRM_FULL_RUN = True before running full mode.")

            active_tasks = list(TASKS)
            if MODE == "full":
                active_tasks = list(dict.fromkeys(TASKS + OPTIONAL_TASKS))

            env_info = collect_environment_info(
                project_path=PROJECT_ROOT,
                official_repo_path=official_repo_path,
                use_gpu_if_available=USE_GPU_IF_AVAILABLE,
            )
            print_environment_info(env_info)
            print("Planned tasks:", active_tasks)
            print("Planned aggregation output path:", AGG_OUTPUT_DIR)

            aggregation_manifest = {
                "start_timestamp": utc_now_iso(),
                "end_timestamp": None,
                "mode": MODE,
                "task_names": active_tasks,
                "seed": SEED,
                "project_git_commit": env_info.get("project_git_commit"),
                "official_repo_git_commit": env_info.get("official_repo_git_commit"),
                "device": env_info.get("device"),
                "gpu_name": env_info.get("gpu_name"),
                "package_versions": env_info.get("package_versions"),
                "chunk_top_k_for_aggregation": CHUNK_TOP_K_FOR_AGGREGATION,
                "doc_top_k_eval": DOC_TOP_K_EVAL,
                "topk_mean_k": TOPK_MEAN_K,
                "softmax_topk_k": SOFTMAX_TOPK_K,
                "softmax_tau": SOFTMAX_TAU,
                "rrf_k": RRF_K,
                "run_bm25": RUN_BM25,
                "run_rrf_fusion": RUN_RRF_FUSION,
                "dependency_workarounds": dependency_workaround_events,
                "output_file_paths": {},
                "metrics": {},
                "notes": [],
            }
            AGG_MANIFEST_PATH = AGG_OUTPUT_DIR / "aggregation_run_manifest.json"
            write_json(AGG_MANIFEST_PATH, aggregation_manifest)
            print(f"Aggregation manifest initialized: {AGG_MANIFEST_PATH}")
            '''
        ),
        markdown("## Official Code Audit"),
        code(
            '''
            audit_path = PROJECT_ROOT / "docs" / "OFFICIAL_CODE_AUDIT.md"
            print(audit_path)
            print(audit_path.read_text(encoding="utf-8")[:3000])
            '''
        ),
        markdown("## Capture Chunk-Level Rankings"),
        code(
            '''
            import importlib
            import os
            from pathlib import Path

            import torch
            from mteb import MTEB
            from transformers import AutoTokenizer

            from latechunk_project.chunk_ranking import patch_official_chunk_ranking_recorder
            from latechunk_project.run_utils import utc_now_iso, write_json

            MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
            CHUNKING_STRATEGY = "fixed"
            CHUNK_SIZE = 256
            N_SENTENCES = 5
            BATCH_SIZE = 1

            captured_tasks = {}
            chunk_ranking_paths = {}

            def _run_task_with_chunk_capture(task_name):
                task_module = importlib.import_module("chunked_pooling.chunked_eval_tasks")
                wrappers = importlib.import_module("chunked_pooling.wrappers")
                task_cls = getattr(task_module, task_name)
                model, has_instructions = wrappers.load_model(MODEL_NAME, None)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
                    model = model.cuda()
                model.eval()

                chunking_args = {
                    "chunk_size": CHUNK_SIZE,
                    "n_sentences": N_SENTENCES,
                    "chunking_strategy": CHUNKING_STRATEGY,
                    "model_has_instructions": has_instructions,
                    "embedding_model_name": MODEL_NAME,
                }
                task = task_cls(
                    chunked_pooling_enabled=True,
                    tokenizer=tokenizer,
                    prune_size=None,
                    truncate_max_length=None,
                    long_late_chunking_embed_size=0,
                    long_late_chunking_overlap_size=256,
                    **chunking_args,
                )
                evaluation = MTEB(
                    tasks=[task],
                    chunked_pooling_enabled=True,
                    tokenizer=tokenizer,
                    prune_size=None,
                    **chunking_args,
                )
                task_dir = AGG_OUTPUT_DIR / "rankings" / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                chunk_ranking_path = task_dir / "chunk_rankings.jsonl"
                result_dir = AGG_OUTPUT_DIR / "official_chunked_pooling_results" / task_name

                previous_cwd = Path.cwd()
                os.chdir(official_repo_path)
                try:
                    with patch_official_chunk_ranking_recorder(
                        chunk_ranking_path,
                        top_k=CHUNK_TOP_K_FOR_AGGREGATION,
                    ):
                        evaluation.run(
                            model,
                            output_folder=str(result_dir),
                            eval_splits=["test"],
                            overwrite_results=True,
                            batch_size=BATCH_SIZE,
                            encode_kwargs={"batch_size": BATCH_SIZE},
                        )
                finally:
                    os.chdir(previous_cwd)

                return task, chunk_ranking_path, result_dir

            if MODE == "smoke":
                print("MODE=smoke: skipping model execution. Use MODE='small' to capture SciFactChunked rankings.")
                aggregation_manifest["notes"].append("Smoke mode skipped chunk ranking capture.")
            else:
                for task_name in active_tasks:
                    print(f"Capturing chunk rankings for {task_name}")
                    task, chunk_path, result_dir = _run_task_with_chunk_capture(task_name)
                    captured_tasks[task_name] = task
                    chunk_ranking_paths[task_name] = chunk_path
                    aggregation_manifest["output_file_paths"].setdefault(task_name, {})
                    aggregation_manifest["output_file_paths"][task_name]["chunk_rankings_jsonl"] = str(chunk_path)
                    aggregation_manifest["output_file_paths"][task_name]["official_chunked_pooling_result_dir"] = str(result_dir)

            aggregation_manifest["last_updated_timestamp"] = utc_now_iso()
            write_json(AGG_MANIFEST_PATH, aggregation_manifest)
            chunk_ranking_paths
            '''
        ),
        markdown("## Evaluate Aggregation Methods"),
        code(
            '''
            import csv
            import json
            from pathlib import Path

            from latechunk_project.analysis import (
                chunks_per_doc_stats,
                compare_per_query,
                correlation,
                per_query_chunk_diagnostics,
            )
            from latechunk_project.bm25_fusion import bm25_rankings, rrf_fuse_rankings
            from latechunk_project.chunk_ranking import load_chunk_rankings_jsonl, save_doc_rankings_jsonl
            from latechunk_project.doc_aggregation import AGGREGATION_METHODS, METHOD_FAMILIES, aggregate_chunk_rankings
            from latechunk_project.evaluation import evaluate_rankings
            from latechunk_project.plotting import plot_delta_histogram, plot_metric_bar, plot_scatter
            from latechunk_project.run_utils import utc_now_iso, write_json

            def _split_or_plain(mapping, split="test"):
                if isinstance(mapping, dict) and split in mapping and isinstance(mapping[split], dict):
                    return mapping[split]
                return mapping

            def _write_csv(path, rows):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not rows:
                    path.write_text("", encoding="utf-8")
                    return path
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return path

            def _jsonable_rows(rows):
                converted = []
                for row in rows:
                    converted.append({
                        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    })
                return converted

            all_summary_rows = []

            if MODE == "smoke":
                print("MODE=smoke: skipping aggregation evaluation.")
            else:
                for task_name in active_tasks:
                    task = captured_tasks[task_name]
                    queries = _split_or_plain(task.queries)
                    corpus = _split_or_plain(task.corpus)
                    qrels = _split_or_plain(task.relevant_docs)
                    chunk_rankings = load_chunk_rankings_jsonl(chunk_ranking_paths[task_name])

                    task_rankings_dir = AGG_OUTPUT_DIR / "rankings" / task_name
                    task_tables_dir = AGG_OUTPUT_DIR / "tables"
                    task_figures_dir = AGG_OUTPUT_DIR / "figures"
                    task_tables_dir.mkdir(parents=True, exist_ok=True)
                    task_figures_dir.mkdir(parents=True, exist_ok=True)

                    method_rankings = {}
                    method_per_query = {}
                    diagnostic_rows = []

                    for method in AGGREGATION_METHODS:
                        rankings, diagnostics = aggregate_chunk_rankings(
                            chunk_rankings,
                            method=method,
                            topk_mean_k=TOPK_MEAN_K,
                            softmax_topk_k=SOFTMAX_TOPK_K,
                            softmax_tau=SOFTMAX_TAU,
                            rrf_k=RRF_K,
                        )
                        method_rankings[method] = rankings
                        save_doc_rankings_jsonl(rankings, task_rankings_dir / f"{method}_doc_ranking.jsonl")
                        metrics, per_query = evaluate_rankings(rankings, qrels, ndcg_k=10, recall_k=100, mrr_k=10)
                        method_per_query[method] = per_query
                        row = {
                            "task": task_name,
                            "method": method,
                            "method_family": METHOD_FAMILIES[method],
                            **metrics,
                            **diagnostics,
                        }
                        all_summary_rows.append(row)
                        diagnostic_rows.append(row)

                    bm25_rankings_by_query = {}
                    if RUN_BM25:
                        bm25_rankings_by_query, bm25_diagnostics = bm25_rankings(corpus, queries, top_k=1000)
                        method_rankings["bm25_only"] = bm25_rankings_by_query
                        save_doc_rankings_jsonl(bm25_rankings_by_query, task_rankings_dir / "bm25_only_doc_ranking.jsonl")
                        metrics, per_query = evaluate_rankings(bm25_rankings_by_query, qrels, ndcg_k=10, recall_k=100, mrr_k=10)
                        method_per_query["bm25_only"] = per_query
                        row = {
                            "task": task_name,
                            "method": "bm25_only",
                            "method_family": METHOD_FAMILIES["bm25_only"],
                            **metrics,
                            **bm25_diagnostics,
                        }
                        all_summary_rows.append(row)
                        diagnostic_rows.append(row)

                    if RUN_BM25 and RUN_RRF_FUSION:
                        fusion_specs = {
                            "late_first_occurrence_plus_bm25_rrf": "first_occurrence",
                            "late_softmax_topk_plus_bm25_rrf": "softmax_topk",
                        }
                        for fusion_method, dense_method in fusion_specs.items():
                            fused = rrf_fuse_rankings(method_rankings[dense_method], bm25_rankings_by_query, rrf_k=RRF_K)
                            method_rankings[fusion_method] = fused
                            save_doc_rankings_jsonl(fused, task_rankings_dir / f"{fusion_method}_doc_ranking.jsonl")
                            metrics, per_query = evaluate_rankings(fused, qrels, ndcg_k=10, recall_k=100, mrr_k=10)
                            method_per_query[fusion_method] = per_query
                            row = {
                                "task": task_name,
                                "method": fusion_method,
                                "method_family": METHOD_FAMILIES[fusion_method],
                                **metrics,
                            }
                            all_summary_rows.append(row)
                            diagnostic_rows.append(row)

                    summary_path = task_tables_dir / f"{task_name}_results_summary.csv"
                    per_query_path = task_tables_dir / f"{task_name}_per_query_metrics.csv"
                    diagnostics_path = task_tables_dir / f"{task_name}_aggregation_diagnostics.csv"
                    bm25_path = task_tables_dir / f"{task_name}_bm25_fusion_results.csv"

                    _write_csv(summary_path, [row for row in all_summary_rows if row["task"] == task_name])
                    per_query_rows = []
                    for method, rows in method_per_query.items():
                        for row in rows:
                            per_query_rows.append({"task": task_name, "method": method, **row})
                    _write_csv(per_query_path, per_query_rows)

                    chunk_diag_rows = per_query_chunk_diagnostics(chunk_rankings, qrels)
                    chunk_stats = chunks_per_doc_stats(chunk_rankings)
                    for row in diagnostic_rows:
                        row.update(chunk_stats)
                    _write_csv(diagnostics_path, diagnostic_rows)

                    if RUN_BM25:
                        _write_csv(bm25_path, [row for row in diagnostic_rows if row["method_family"] in {"lexical", "hybrid"}])

                    proposed = "softmax_topk" if "softmax_topk" in method_per_query else "topk_mean"
                    comparison_rows = compare_per_query(
                        method_per_query["first_occurrence"],
                        method_per_query[proposed],
                        method_rankings["first_occurrence"],
                        method_rankings[proposed],
                        queries,
                        qrels,
                        proposed_method=proposed,
                    )
                    by_query_chunk_diag = {row["query_id"]: row for row in chunk_diag_rows}
                    for row in comparison_rows:
                        row.update({
                            "avg_chunks_per_retrieved_doc": by_query_chunk_diag.get(row["query_id"], {}).get("avg_chunks_per_retrieved_doc", 0.0),
                            "relevant_chunk_counts": by_query_chunk_diag.get(row["query_id"], {}).get("relevant_chunk_counts", {}),
                        })

                    improved_rows = sorted([row for row in comparison_rows if row["delta"] > 0], key=lambda row: -row["delta"])
                    degraded_rows = sorted([row for row in comparison_rows if row["delta"] < 0], key=lambda row: row["delta"])
                    _write_csv(task_tables_dir / f"{task_name}_improved_queries.csv", _jsonable_rows(improved_rows))
                    _write_csv(task_tables_dir / f"{task_name}_degraded_queries.csv", _jsonable_rows(degraded_rows))

                    xs = [float(row.get("avg_chunks_per_retrieved_doc", 0.0)) for row in comparison_rows]
                    ys = [float(row["delta"]) for row in comparison_rows]
                    corr = correlation(xs, ys)
                    print(f"{task_name}: correlation between chunks/doc and {proposed} improvement = {corr:.4f}")

                    task_summary_rows = [row for row in all_summary_rows if row["task"] == task_name]
                    plot_metric_bar(task_summary_rows, "ndcg_at_10", task_figures_dir / f"{task_name}_method_comparison_ndcg10.png")
                    plot_metric_bar(task_summary_rows, "recall_at_100", task_figures_dir / f"{task_name}_method_comparison_recall100.png")
                    plot_delta_histogram(ys, task_figures_dir / f"{task_name}_per_query_delta_histogram.png")
                    plot_scatter(xs, ys, "avg chunks per retrieved doc", "nDCG@10 delta", task_figures_dir / f"{task_name}_chunks_per_doc_vs_improvement.png")

                    aggregation_manifest["metrics"][task_name] = {
                        row["method"]: row for row in task_summary_rows
                    }
                    aggregation_manifest["output_file_paths"].setdefault(task_name, {})
                    aggregation_manifest["output_file_paths"][task_name].update({
                        "results_summary_csv": str(summary_path),
                        "per_query_metrics_csv": str(per_query_path),
                        "aggregation_diagnostics_csv": str(diagnostics_path),
                        "bm25_fusion_results_csv": str(bm25_path),
                    })

            aggregation_manifest["end_timestamp"] = utc_now_iso()
            write_json(AGG_MANIFEST_PATH, aggregation_manifest)
            all_summary_rows
            '''
        ),
        markdown("## Results Reminder"),
        markdown(
            """
            Do not report improvements unless this notebook actually executed and
            saved the result artifacts. BM25/RRF rows are hybrid dense + lexical
            baselines, not pure Late Chunking aggregation improvements.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
