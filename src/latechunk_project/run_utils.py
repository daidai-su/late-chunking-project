"""Run bookkeeping helpers for Colab baseline jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    """Create the expected output artifact directories."""
    base = Path(output_dir)
    dirs = {
        "base": base,
        "logs": base / "logs",
        "tables": base / "tables",
        "predictions": base / "predictions",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def save_run_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Save run_manifest.json under the configured output directory."""
    return write_json(Path(output_dir) / "run_manifest.json", manifest)


def as_posix_str(path: str | Path | None) -> str | None:
    """Convert paths to JSON-friendly strings while preserving missing values."""
    if path is None:
        return None
    return Path(path).as_posix()

