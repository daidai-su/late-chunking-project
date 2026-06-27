"""Parse retrieval metrics from official baseline logs when possible."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


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

