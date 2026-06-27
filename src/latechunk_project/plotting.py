"""Matplotlib plotting helpers for aggregation experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def plot_metric_bar(rows: Sequence[Mapping[str, object]], metric_key: str, output_path: str | Path) -> Path:
    """Create one method comparison bar chart."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in rows]
    values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]
    plt.figure(figsize=(max(8, len(labels) * 1.2), 4))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_delta_histogram(deltas: Sequence[float], output_path: str | Path) -> Path:
    """Create a per-query delta histogram."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist([float(delta) for delta in deltas], bins=20)
    plt.xlabel("nDCG@10 delta")
    plt.ylabel("queries")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_scatter(xs: Sequence[float], ys: Sequence[float], xlabel: str, ylabel: str, output_path: str | Path) -> Path:
    """Create one scatter plot."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter([float(x) for x in xs], [float(y) for y in ys])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

