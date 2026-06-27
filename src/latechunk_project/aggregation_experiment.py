"""Subprocess runner for Phase B aggregation experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _split_or_plain(mapping: dict[str, Any], split: str = "test") -> dict[str, Any]:
    if isinstance(mapping, dict) and split in mapping and isinstance(mapping[split], dict):
        return mapping[split]
    return mapping


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        converted.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return converted


def _prepare_paths(output_dir: str | Path, aggregation_subdir: str) -> Path:
    aggregation_dir = Path(output_dir) / aggregation_subdir
    for name in ["rankings", "tables", "figures", "logs"]:
        (aggregation_dir / name).mkdir(parents=True, exist_ok=True)
    return aggregation_dir


def _capture_task_rankings(
    *,
    task_name: str,
    official_repo_path: Path,
    aggregation_dir: Path,
    model_name: str,
    use_gpu_if_available: bool,
    chunking_strategy: str,
    chunk_size: int,
    n_sentences: int,
    batch_size: int,
    chunk_top_k: int,
) -> tuple[Any, Path, Path]:
    import torch
    from mteb import MTEB
    from transformers import AutoTokenizer

    from latechunk_project.chunk_ranking import patch_official_chunk_ranking_recorder

    task_module = importlib.import_module("chunked_pooling.chunked_eval_tasks")
    wrappers = importlib.import_module("chunked_pooling.wrappers")
    task_cls = getattr(task_module, task_name)
    model, has_instructions = wrappers.load_model(model_name, None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if use_gpu_if_available and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    chunking_args = {
        "chunk_size": chunk_size,
        "n_sentences": n_sentences,
        "chunking_strategy": chunking_strategy,
        "model_has_instructions": has_instructions,
        "embedding_model_name": model_name,
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
    task_dir = aggregation_dir / "rankings" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    chunk_ranking_path = task_dir / "chunk_rankings.jsonl"
    result_dir = aggregation_dir / "official_chunked_pooling_results" / task_name

    previous_cwd = Path.cwd()
    os.chdir(official_repo_path)
    try:
        with patch_official_chunk_ranking_recorder(chunk_ranking_path, top_k=chunk_top_k):
            evaluation.run(
                model,
                output_folder=str(result_dir),
                eval_splits=["test"],
                overwrite_results=True,
                batch_size=batch_size,
                encode_kwargs={"batch_size": batch_size},
            )
    finally:
        os.chdir(previous_cwd)

    return task, chunk_ranking_path, result_dir


def run_aggregation_experiment(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    official_repo_path = Path(config["official_repo_path"])
    output_dir = Path(config["output_dir"])
    aggregation_dir = _prepare_paths(output_dir, config["agg_output_subdir"])

    os.chdir(project_root)
    project_src = str(project_root / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    official_src = str(official_repo_path)
    if official_src not in sys.path:
        sys.path.insert(0, official_src)

    from latechunk_project.analysis import (
        chunks_per_doc_stats,
        compare_per_query,
        correlation,
        per_query_chunk_diagnostics,
    )
    from latechunk_project.bm25_fusion import bm25_rankings, rrf_fuse_rankings
    from latechunk_project.chunk_ranking import load_chunk_rankings_jsonl, save_doc_rankings_jsonl
    from latechunk_project.dependency_workarounds import (
        ensure_numpy_stack_healthy,
        remove_optional_media_dependencies,
    )
    from latechunk_project.doc_aggregation import (
        AGGREGATION_METHODS,
        METHOD_FAMILIES,
        aggregate_chunk_rankings,
    )
    from latechunk_project.env_utils import collect_environment_info, print_environment_info
    from latechunk_project.evaluation import evaluate_rankings
    from latechunk_project.plotting import plot_delta_histogram, plot_metric_bar, plot_scatter
    from latechunk_project.run_utils import utc_now_iso, write_json

    numpy_stack_status = ensure_numpy_stack_healthy()
    print("NumPy stack check in experiment subprocess:", numpy_stack_status)
    dependency_workaround_events = list(config.get("dependency_workaround_events", []))
    dependency_workaround_events.extend(remove_optional_media_dependencies())
    print("Optional media dependency checks in experiment subprocess:", dependency_workaround_events)

    mode = config["mode"]
    if mode == "full" and not config.get("confirm_full_run", False):
        raise SystemExit("Set CONFIRM_FULL_RUN = True before running full mode.")

    active_tasks = list(config["tasks"])
    if mode == "full":
        active_tasks = list(dict.fromkeys(config["tasks"] + config.get("optional_tasks", [])))

    env_info = collect_environment_info(
        project_path=project_root,
        official_repo_path=official_repo_path,
        use_gpu_if_available=config["use_gpu_if_available"],
    )
    print_environment_info(env_info)
    print("Planned tasks:", active_tasks)
    print("Planned aggregation output path:", aggregation_dir)

    aggregation_manifest = {
        "start_timestamp": utc_now_iso(),
        "end_timestamp": None,
        "mode": mode,
        "task_names": active_tasks,
        "seed": config["seed"],
        "project_git_commit": env_info.get("project_git_commit"),
        "official_repo_git_commit": env_info.get("official_repo_git_commit"),
        "device": env_info.get("device"),
        "gpu_name": env_info.get("gpu_name"),
        "package_versions": env_info.get("package_versions"),
        "chunk_top_k_for_aggregation": config["chunk_top_k_for_aggregation"],
        "doc_top_k_eval": config["doc_top_k_eval"],
        "topk_mean_k": config["topk_mean_k"],
        "softmax_topk_k": config["softmax_topk_k"],
        "softmax_tau": config["softmax_tau"],
        "rrf_k": config["rrf_k"],
        "run_bm25": config["run_bm25"],
        "run_rrf_fusion": config["run_rrf_fusion"],
        "dependency_workarounds": dependency_workaround_events,
        "numpy_stack_status": numpy_stack_status,
        "output_file_paths": {},
        "metrics": {},
        "notes": [],
    }
    manifest_path = aggregation_dir / "aggregation_run_manifest.json"
    write_json(manifest_path, aggregation_manifest)
    print(f"Aggregation manifest initialized: {manifest_path}")

    audit_path = project_root / "docs" / "OFFICIAL_CODE_AUDIT.md"
    if audit_path.exists():
        print(audit_path)
        print(audit_path.read_text(encoding="utf-8")[:3000])

    captured_tasks: dict[str, Any] = {}
    chunk_ranking_paths: dict[str, Path] = {}
    if mode == "smoke":
        print("MODE=smoke: skipping model execution. Use MODE='small' to capture rankings.")
        aggregation_manifest["notes"].append("Smoke mode skipped chunk ranking capture.")
    else:
        for task_name in active_tasks:
            print(f"Capturing chunk rankings for {task_name}")
            task, chunk_path, result_dir = _capture_task_rankings(
                task_name=task_name,
                official_repo_path=official_repo_path,
                aggregation_dir=aggregation_dir,
                model_name=config["model_name"],
                use_gpu_if_available=config["use_gpu_if_available"],
                chunking_strategy=config["chunking_strategy"],
                chunk_size=config["chunk_size"],
                n_sentences=config["n_sentences"],
                batch_size=config["batch_size"],
                chunk_top_k=config["chunk_top_k_for_aggregation"],
            )
            captured_tasks[task_name] = task
            chunk_ranking_paths[task_name] = chunk_path
            aggregation_manifest["output_file_paths"].setdefault(task_name, {})
            aggregation_manifest["output_file_paths"][task_name]["chunk_rankings_jsonl"] = str(chunk_path)
            aggregation_manifest["output_file_paths"][task_name][
                "official_chunked_pooling_result_dir"
            ] = str(result_dir)

    aggregation_manifest["last_updated_timestamp"] = utc_now_iso()
    write_json(manifest_path, aggregation_manifest)

    all_summary_rows: list[dict[str, Any]] = []
    if mode == "smoke":
        print("MODE=smoke: skipping aggregation evaluation.")
    else:
        for task_name in active_tasks:
            task = captured_tasks[task_name]
            queries = _split_or_plain(task.queries)
            corpus = _split_or_plain(task.corpus)
            qrels = _split_or_plain(task.relevant_docs)
            chunk_rankings = load_chunk_rankings_jsonl(chunk_ranking_paths[task_name])

            task_rankings_dir = aggregation_dir / "rankings" / task_name
            task_tables_dir = aggregation_dir / "tables"
            task_figures_dir = aggregation_dir / "figures"
            task_tables_dir.mkdir(parents=True, exist_ok=True)
            task_figures_dir.mkdir(parents=True, exist_ok=True)

            method_rankings = {}
            method_per_query = {}
            diagnostic_rows = []

            for method in AGGREGATION_METHODS:
                rankings, diagnostics = aggregate_chunk_rankings(
                    chunk_rankings,
                    method=method,
                    topk_mean_k=config["topk_mean_k"],
                    softmax_topk_k=config["softmax_topk_k"],
                    softmax_tau=config["softmax_tau"],
                    rrf_k=config["rrf_k"],
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
            if config["run_bm25"]:
                bm25_rankings_by_query, bm25_diagnostics = bm25_rankings(corpus, queries, top_k=1000)
                method_rankings["bm25_only"] = bm25_rankings_by_query
                save_doc_rankings_jsonl(
                    bm25_rankings_by_query,
                    task_rankings_dir / "bm25_only_doc_ranking.jsonl",
                )
                metrics, per_query = evaluate_rankings(
                    bm25_rankings_by_query,
                    qrels,
                    ndcg_k=10,
                    recall_k=100,
                    mrr_k=10,
                )
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

            if config["run_bm25"] and config["run_rrf_fusion"]:
                fusion_specs = {
                    "late_first_occurrence_plus_bm25_rrf": "first_occurrence",
                    "late_softmax_topk_plus_bm25_rrf": "softmax_topk",
                }
                for fusion_method, dense_method in fusion_specs.items():
                    fused = rrf_fuse_rankings(
                        method_rankings[dense_method],
                        bm25_rankings_by_query,
                        rrf_k=config["rrf_k"],
                    )
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

            if config["run_bm25"]:
                _write_csv(
                    bm25_path,
                    [row for row in diagnostic_rows if row["method_family"] in {"lexical", "hybrid"}],
                )

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
                row.update(
                    {
                        "avg_chunks_per_retrieved_doc": by_query_chunk_diag.get(row["query_id"], {}).get(
                            "avg_chunks_per_retrieved_doc",
                            0.0,
                        ),
                        "relevant_chunk_counts": by_query_chunk_diag.get(row["query_id"], {}).get(
                            "relevant_chunk_counts",
                            {},
                        ),
                    }
                )

            improved_rows = sorted(
                [row for row in comparison_rows if row["delta"] > 0],
                key=lambda row: -row["delta"],
            )
            degraded_rows = sorted(
                [row for row in comparison_rows if row["delta"] < 0],
                key=lambda row: row["delta"],
            )
            _write_csv(task_tables_dir / f"{task_name}_improved_queries.csv", _jsonable_rows(improved_rows))
            _write_csv(task_tables_dir / f"{task_name}_degraded_queries.csv", _jsonable_rows(degraded_rows))

            xs = [float(row.get("avg_chunks_per_retrieved_doc", 0.0)) for row in comparison_rows]
            ys = [float(row["delta"]) for row in comparison_rows]
            corr = correlation(xs, ys)
            print(f"{task_name}: correlation between chunks/doc and {proposed} improvement = {corr:.4f}")

            task_summary_rows = [row for row in all_summary_rows if row["task"] == task_name]
            plot_metric_bar(
                task_summary_rows,
                "ndcg_at_10",
                task_figures_dir / f"{task_name}_method_comparison_ndcg10.png",
            )
            plot_metric_bar(
                task_summary_rows,
                "recall_at_100",
                task_figures_dir / f"{task_name}_method_comparison_recall100.png",
            )
            plot_delta_histogram(ys, task_figures_dir / f"{task_name}_per_query_delta_histogram.png")
            plot_scatter(
                xs,
                ys,
                "avg chunks per retrieved doc",
                "nDCG@10 delta",
                task_figures_dir / f"{task_name}_chunks_per_doc_vs_improvement.png",
            )

            aggregation_manifest["metrics"][task_name] = {row["method"]: row for row in task_summary_rows}
            aggregation_manifest["output_file_paths"].setdefault(task_name, {})
            aggregation_manifest["output_file_paths"][task_name].update(
                {
                    "results_summary_csv": str(summary_path),
                    "per_query_metrics_csv": str(per_query_path),
                    "aggregation_diagnostics_csv": str(diagnostics_path),
                    "bm25_fusion_results_csv": str(bm25_path),
                }
            )

    summary_all_path = aggregation_dir / "tables" / "aggregation_results_summary.csv"
    _write_csv(summary_all_path, all_summary_rows)
    aggregation_manifest["output_file_paths"]["aggregation_results_summary_csv"] = str(summary_all_path)
    aggregation_manifest["end_timestamp"] = utc_now_iso()
    write_json(manifest_path, aggregation_manifest)
    print(f"Aggregation manifest saved: {manifest_path}")
    print(f"Aggregation summary saved: {summary_all_path}")
    return aggregation_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config_json)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_aggregation_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
