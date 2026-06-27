"""Parse retrieval metrics from official baseline logs and result files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_KEY_VALUE_PATTERNS = [
    re.compile(
        rf"(?P<name>nDCG|NDCG|ndcg|Recall|recall|MRR|mrr|MAP|map|Precision|precision)"
        rf"\s*(?:@|[_ -]?at[_ -]?)?\s*(?P<k>\d+)?\s*[:=]\s*(?P<value>{_NUMBER})"
    ),
    re.compile(
        rf"['\"](?P<name>(?:nDCG|NDCG|ndcg|Recall|recall|MRR|mrr|MAP|map|Precision|precision)"
        rf"(?:@|[_-]at[_-]?)?\d*)['\"]\s*:\s*(?P<value>{_NUMBER})"
    ),
]


def normalize_metric_name(raw_name: str, k: str | None = None) -> str:
    """Normalize common metric spellings to names such as nDCG@10."""
    cleaned = raw_name.strip().strip("'\"")
    lowered = cleaned.lower().replace("-", "_").replace(" ", "")
    match = re.match(r"^(ndcg|recall|mrr|map|precision)(?:@|_?at_?)?(\d+)?$", lowered)
    if not match:
        return cleaned

    metric, embedded_k = match.groups()
    final_k = k or embedded_k
    canonical = {
        "ndcg": "nDCG",
        "recall": "Recall",
        "mrr": "MRR",
        "map": "MAP",
        "precision": "Precision",
    }[metric]
    return f"{canonical}@{final_k}" if final_k else canonical


def parse_official_metrics(
    output_text: str,
    wanted_metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Parse metric values from official stdout/stderr text.

    The official scripts may change their final print format. This parser
    deliberately accepts several simple key-value formats and returns an empty
    dict rather than guessing when no metric is visible.
    """
    wanted = {normalize_metric_name(name) for name in wanted_metrics} if wanted_metrics else None
    metrics: dict[str, float] = {}
    for pattern in _KEY_VALUE_PATTERNS:
        for match in pattern.finditer(output_text):
            name = normalize_metric_name(match.group("name"), match.groupdict().get("k"))
            if wanted is not None and name not in wanted:
                continue
            metrics[name] = float(match.group("value"))
    return metrics


def parse_metrics_from_file(path: str | Path, wanted_metrics: Iterable[str] | None = None) -> dict[str, float]:
    """Read a log file and parse visible metric values."""
    log_path = Path(path)
    if not log_path.exists():
        return {}
    return parse_official_metrics(log_path.read_text(encoding="utf-8", errors="replace"), wanted_metrics)


def parse_mteb_result_json(path: str | Path) -> dict[str, Any]:
    """Parse metrics from an MTEB JSON result file written by the official repo."""
    result_path = Path(path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    test_scores = data.get("scores", {}).get("test", [])
    if not test_scores:
        return {
            "path": str(result_path),
            "task_name": data.get("task_name"),
            "dataset_revision": data.get("dataset_revision"),
            "mteb_version": data.get("mteb_version"),
            "evaluation_time": data.get("evaluation_time"),
            "metrics": {},
            "raw_score": {},
        }

    raw_score = dict(test_scores[0])
    metrics = {
        normalize_metric_name(key): float(value)
        for key, value in raw_score.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        "path": str(result_path),
        "task_name": data.get("task_name"),
        "dataset_revision": data.get("dataset_revision"),
        "mteb_version": data.get("mteb_version"),
        "evaluation_time": data.get("evaluation_time"),
        "metrics": metrics,
        "raw_score": raw_score,
    }


def method_name_from_result_dir(path: str | Path) -> str:
    """Convert official result directories to compact method names."""
    result_path = Path(path)
    for parent in [result_path.parent, *result_path.parents]:
        name = parent.name
        if name.startswith("results-"):
            return name[len("results-") :].replace("-", "_")
    return result_path.parent.name.replace("-", "_")


def find_mteb_result_jsons(
    official_repo_dir: str | Path,
    task_names: Iterable[str],
) -> dict[str, dict[str, Path]]:
    """Find official MTEB JSON result files keyed by task and method."""
    repo_dir = Path(official_repo_dir)
    found: dict[str, dict[str, Path]] = {}
    for task_name in task_names:
        task_results: dict[str, Path] = {}
        for path in repo_dir.glob(f"results-*/*/*/{task_name}.json"):
            task_results[method_name_from_result_dir(path)] = path
        if task_results:
            found[task_name] = dict(sorted(task_results.items()))
    return found


def summarize_mteb_result_jsons(
    official_repo_dir: str | Path,
    task_names: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Create CSV-friendly rows and manifest-friendly metrics from result JSONs."""
    result_paths = find_mteb_result_jsons(official_repo_dir, task_names)
    rows: list[dict[str, Any]] = []
    manifest_metrics: dict[str, dict[str, dict[str, Any]]] = {}

    for task_name, method_paths in result_paths.items():
        manifest_metrics[task_name] = {}
        for method, path in method_paths.items():
            parsed = parse_mteb_result_json(path)
            metrics = parsed["metrics"]
            raw_score = parsed["raw_score"]
            manifest_metrics[task_name][method] = raw_score
            rows.append(
                {
                    "task": parsed["task_name"] or task_name,
                    "method": method,
                    "main_score": metrics.get("main_score"),
                    "ndcg_at_10": metrics.get("nDCG@10"),
                    "recall_at_10": metrics.get("Recall@10"),
                    "map_at_10": metrics.get("MAP@10"),
                    "mrr_at_10": metrics.get("MRR@10"),
                    "evaluation_time": parsed["evaluation_time"],
                    "result_json": parsed["path"],
                }
            )

    return rows, manifest_metrics
