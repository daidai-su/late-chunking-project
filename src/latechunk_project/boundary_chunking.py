"""Boundary-aware token span utilities for optional Late Chunking experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


BOUNDARY_CHUNKING_METHODS = (
    "fixed_256_tokens",
    "sentence_boundary_approx",
    "paragraph_boundary_approx",
    "overlap_fixed",
)


@dataclass(frozen=True)
class ChunkSpan:
    """Half-open token span."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def whitespace_token_offsets(text: str) -> list[tuple[int, int]]:
    """Return simple non-whitespace token offsets for tests and diagnostics."""
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _clean_offsets(token_offsets: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = [(int(start), int(end)) for start, end in token_offsets if int(end) > int(start)]
    return sorted(cleaned, key=lambda item: (item[0], item[1]))


def _fixed_interval_spans(start: int, end: int, max_chunk_tokens: int) -> list[ChunkSpan]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    spans: list[ChunkSpan] = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + max_chunk_tokens, end)
        spans.append(ChunkSpan(cursor, next_cursor))
        cursor = next_cursor
    return spans


def fixed_token_spans(num_tokens: int, max_chunk_tokens: int = 256) -> list[ChunkSpan]:
    """Return non-overlapping fixed-width token chunks."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    return _fixed_interval_spans(0, num_tokens, max_chunk_tokens)


def overlap_token_spans(
    num_tokens: int,
    max_chunk_tokens: int = 256,
    overlap_tokens: int = 64,
) -> list[ChunkSpan]:
    """Return deterministic fixed-width token chunks with overlap."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")
    if num_tokens == 0:
        return []

    spans: list[ChunkSpan] = []
    step = max_chunk_tokens - overlap_tokens
    start = 0
    while start < num_tokens:
        end = min(start + max_chunk_tokens, num_tokens)
        spans.append(ChunkSpan(start, end))
        if end == num_tokens:
            break
        start += step
    return spans


def sentence_unit_spans(text: str, token_offsets: Sequence[tuple[int, int]]) -> list[ChunkSpan]:
    """Split token offsets by simple punctuation-based sentence boundaries."""
    offsets = _clean_offsets(token_offsets)
    if not offsets:
        return []

    spans: list[ChunkSpan] = []
    start = 0
    for index, (char_start, char_end) in enumerate(offsets):
        token_text = text[char_start:char_end].strip()
        if re.search(r"[.!?]+[\"')\]]*$", token_text):
            spans.append(ChunkSpan(start, index + 1))
            start = index + 1
    if start < len(offsets):
        spans.append(ChunkSpan(start, len(offsets)))
    return [span for span in spans if span.length > 0]


def _paragraph_char_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    cursor = 0
    saw_blank_line = False

    for line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end
        if line.strip():
            if paragraph_start is None:
                paragraph_start = line_start
            paragraph_end = line_end
        else:
            saw_blank_line = True
            if paragraph_start is not None and paragraph_end is not None:
                ranges.append((paragraph_start, paragraph_end))
            paragraph_start = None
            paragraph_end = None

    if paragraph_start is not None and paragraph_end is not None:
        ranges.append((paragraph_start, paragraph_end))

    if not ranges and text.strip():
        ranges.append((0, len(text)))
    if not saw_blank_line and len(ranges) <= 1:
        return []
    return ranges


def _char_ranges_to_token_spans(
    char_ranges: Sequence[tuple[int, int]],
    token_offsets: Sequence[tuple[int, int]],
) -> list[ChunkSpan]:
    offsets = _clean_offsets(token_offsets)
    spans: list[ChunkSpan] = []
    cursor = 0
    for range_start, range_end in char_ranges:
        while cursor < len(offsets) and offsets[cursor][1] <= range_start:
            cursor += 1
        token_start = cursor
        while cursor < len(offsets) and offsets[cursor][0] < range_end:
            cursor += 1
        token_end = cursor
        if token_end > token_start:
            spans.append(ChunkSpan(token_start, token_end))
    return spans


def paragraph_unit_spans(text: str, token_offsets: Sequence[tuple[int, int]]) -> list[ChunkSpan]:
    """Split token offsets by paragraph markers, falling back to sentences."""
    paragraph_ranges = _paragraph_char_ranges(text)
    if not paragraph_ranges:
        return sentence_unit_spans(text, token_offsets)

    paragraph_spans = _char_ranges_to_token_spans(paragraph_ranges, token_offsets)
    if not paragraph_spans:
        return sentence_unit_spans(text, token_offsets)
    return _fill_token_gaps(paragraph_spans, len(_clean_offsets(token_offsets)))


def _fill_token_gaps(unit_spans: Sequence[ChunkSpan], num_tokens: int) -> list[ChunkSpan]:
    filled: list[ChunkSpan] = []
    cursor = 0
    for span in sorted(unit_spans, key=lambda item: (item.start, item.end)):
        if span.start > cursor:
            filled.append(ChunkSpan(cursor, span.start))
        if span.end > span.start:
            filled.append(ChunkSpan(max(span.start, cursor), span.end))
            cursor = max(cursor, span.end)
    if cursor < num_tokens:
        filled.append(ChunkSpan(cursor, num_tokens))
    return [span for span in filled if span.length > 0]


def pack_units_to_max_tokens(
    unit_spans: Sequence[ChunkSpan],
    max_chunk_tokens: int = 256,
) -> list[ChunkSpan]:
    """Pack sentence or paragraph units into chunks bounded by max tokens."""
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    if not unit_spans:
        return []

    chunks: list[ChunkSpan] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None and current_end > current_start:
            chunks.append(ChunkSpan(current_start, current_end))
        current_start = None
        current_end = None

    for unit in sorted(unit_spans, key=lambda item: (item.start, item.end)):
        if unit.length <= 0:
            continue
        if unit.length > max_chunk_tokens:
            flush_current()
            chunks.extend(_fixed_interval_spans(unit.start, unit.end, max_chunk_tokens))
            continue
        if current_start is None:
            current_start = unit.start
            current_end = unit.end
            continue
        assert current_end is not None
        if unit.end - current_start <= max_chunk_tokens:
            current_end = unit.end
        else:
            flush_current()
            current_start = unit.start
            current_end = unit.end
    flush_current()
    return chunks


def boundary_chunk_spans(
    text: str,
    token_offsets: Sequence[tuple[int, int]],
    method: str,
    max_chunk_tokens: int = 256,
    overlap_tokens: int = 64,
) -> list[ChunkSpan]:
    """Create token spans for a boundary-aware chunking method."""
    offsets = _clean_offsets(token_offsets)
    num_tokens = len(offsets)
    if method not in BOUNDARY_CHUNKING_METHODS:
        raise ValueError(f"Unsupported boundary chunking method: {method}")
    if method == "fixed_256_tokens":
        spans = fixed_token_spans(num_tokens, max_chunk_tokens=max_chunk_tokens)
    elif method == "overlap_fixed":
        spans = overlap_token_spans(
            num_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
    elif method == "sentence_boundary_approx":
        units = sentence_unit_spans(text, offsets)
        spans = pack_units_to_max_tokens(_fill_token_gaps(units, num_tokens), max_chunk_tokens)
    else:
        units = paragraph_unit_spans(text, offsets)
        spans = pack_units_to_max_tokens(_fill_token_gaps(units, num_tokens), max_chunk_tokens)
    return [span for span in spans if span.length > 0]


def chunk_text_by_whitespace(
    text: str,
    method: str,
    max_chunk_tokens: int = 256,
    overlap_tokens: int = 64,
) -> list[ChunkSpan]:
    """Chunk text using simple whitespace tokenization."""
    return boundary_chunk_spans(
        text,
        whitespace_token_offsets(text),
        method=method,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )


def validate_non_overlapping_coverage(spans: Sequence[ChunkSpan], num_tokens: int) -> bool:
    """Return True when spans cover every token exactly once, in order."""
    cursor = 0
    for span in spans:
        if span.start != cursor or span.end <= span.start:
            return False
        cursor = span.end
    return cursor == num_tokens


def validate_overlapping_coverage(spans: Sequence[ChunkSpan], num_tokens: int) -> bool:
    """Return True when sorted overlapping spans cover all tokens in order."""
    if num_tokens == 0:
        return list(spans) == []
    if not spans or spans[0].start != 0:
        return False
    covered_until = 0
    for span in spans:
        if span.start > covered_until or span.end <= span.start:
            return False
        covered_until = max(covered_until, span.end)
    return covered_until == num_tokens


def chunk_statistics(spans: Sequence[ChunkSpan], embedding_dim: int = 512, dtype_bytes: int = 4) -> dict[str, float]:
    """Compute chunk count, average length, and embedding memory estimate."""
    lengths = [span.length for span in spans]
    num_chunks = len(lengths)
    return {
        "num_chunks": float(num_chunks),
        "avg_chunk_length": sum(lengths) / num_chunks if num_chunks else 0.0,
        "max_chunk_length": float(max(lengths)) if lengths else 0.0,
        "estimated_embedding_memory_mb": (num_chunks * embedding_dim * dtype_bytes) / (1024 * 1024),
    }


def _tokenizer_offsets(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    return _clean_offsets(encoded.get("offset_mapping", []))


@contextmanager
def patch_official_boundary_chunker(
    max_chunk_tokens: int = 256,
    overlap_tokens: int = 64,
) -> Iterator[None]:
    """Temporarily add boundary-aware methods to the official Chunker."""
    chunking_module = importlib.import_module("chunked_pooling.chunking")
    chunker_cls = chunking_module.Chunker
    original_init = chunker_cls.__init__
    original_chunk = chunker_cls.chunk
    original_strategies = getattr(chunking_module, "CHUNKING_STRATEGIES", None)

    if isinstance(original_strategies, list):
        chunking_module.CHUNKING_STRATEGIES = list(
            dict.fromkeys([*original_strategies, *BOUNDARY_CHUNKING_METHODS])
        )

    def patched_init(self: Any, chunking_strategy: str = "fixed") -> None:
        if chunking_strategy in BOUNDARY_CHUNKING_METHODS:
            self.chunking_strategy = chunking_strategy
            return
        original_init(self, chunking_strategy)

    def patched_chunk(self: Any, text: str, tokenizer: Any, *args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        strategy = kwargs.get("chunking_strategy") or getattr(self, "chunking_strategy", None)
        if strategy in BOUNDARY_CHUNKING_METHODS:
            offsets = _tokenizer_offsets(text, tokenizer)
            spans = boundary_chunk_spans(
                text,
                offsets,
                method=strategy,
                max_chunk_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
            return [(span.start, span.end) for span in spans]
        return original_chunk(self, text, tokenizer, *args, **kwargs)

    chunker_cls.__init__ = patched_init
    chunker_cls.chunk = patched_chunk
    try:
        yield
    finally:
        chunker_cls.__init__ = original_init
        chunker_cls.chunk = original_chunk
        if isinstance(original_strategies, list):
            chunking_module.CHUNKING_STRATEGIES = original_strategies


def _extract_text(document: Any) -> str:
    if isinstance(document, dict):
        return str(document.get("text") or document.get("title") or "")
    return str(document)


def _plain_split(mapping: Any, split: str = "test") -> Any:
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


def _find_task_result_json(output_dir: Path, task_name: str) -> Path | None:
    matches = sorted(output_dir.glob(f"**/{task_name}.json"))
    return matches[-1] if matches else None


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _load_result_metrics(result_json: Path | None) -> dict[str, Any]:
    if result_json is None or not result_json.exists():
        return {}
    from latechunk_project.result_parsing import parse_mteb_result_json

    parsed = parse_mteb_result_json(result_json)
    raw = parsed.get("raw_score", {})
    metrics = parsed.get("metrics", {})
    return {
        "result_json": str(result_json),
        "ndcg_at_10": _metric(metrics, "nDCG@10", "ndcg_at_10") or _metric(raw, "ndcg_at_10"),
        "recall_at_100": _metric(metrics, "Recall@100", "recall_at_100") or _metric(raw, "recall_at_100"),
        "mrr_at_10": _metric(metrics, "MRR@10", "mrr_at_10") or _metric(raw, "mrr_at_10"),
        "evaluation_time": parsed.get("evaluation_time"),
    }


def _corpus_chunk_stats(
    corpus: dict[str, Any],
    tokenizer: Any,
    method: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
    embedding_dim: int,
) -> dict[str, float]:
    all_lengths: list[int] = []
    for document in corpus.values():
        text = _extract_text(document)
        offsets = _tokenizer_offsets(text, tokenizer)
        spans = boundary_chunk_spans(
            text,
            offsets,
            method=method,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_lengths.extend(span.length for span in spans)
    num_chunks = len(all_lengths)
    return {
        "num_chunks": float(num_chunks),
        "avg_chunk_length": sum(all_lengths) / num_chunks if num_chunks else 0.0,
        "max_chunk_length": float(max(all_lengths)) if all_lengths else 0.0,
        "estimated_embedding_memory_mb": (num_chunks * embedding_dim * 4) / (1024 * 1024),
    }


def run_boundary_chunking_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the optional boundary-aware chunking experiment."""
    project_root = Path(config["project_root"])
    official_repo_path = Path(config["official_repo_path"])
    output_dir = Path(config["output_dir"])
    boundary_dir = output_dir / config.get("boundary_output_subdir", "boundary_chunking_experiment")
    for name in ["tables", "logs", "results"]:
        (boundary_dir / name).mkdir(parents=True, exist_ok=True)

    os.chdir(project_root)
    project_src = str(project_root / "src")
    official_src = str(official_repo_path)
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    if official_src not in sys.path:
        sys.path.insert(0, official_src)

    from latechunk_project.dependency_workarounds import (
        ensure_numpy_stack_healthy,
        remove_optional_media_dependencies,
    )
    from latechunk_project.env_utils import collect_environment_info
    from latechunk_project.run_utils import utc_now_iso, write_json

    numpy_status = ensure_numpy_stack_healthy()
    dependency_workarounds = remove_optional_media_dependencies()

    from mteb import MTEB
    from transformers import AutoTokenizer

    task_names = list(config.get("task_names") or ["SciFactChunked"])
    methods = list(config.get("chunking_methods") or BOUNDARY_CHUNKING_METHODS)
    max_chunk_tokens = int(config.get("max_chunk_tokens", 256))
    overlap_tokens = int(config.get("overlap_tokens", 64))
    model_name = config.get("model_name", "jinaai/jina-embeddings-v2-small-en")
    embedding_dim = int(config.get("embedding_dim", 512))
    batch_size = int(config.get("batch_size", 1))
    use_gpu = bool(config.get("use_gpu_if_available", True))

    env_info = collect_environment_info(
        project_path=project_root,
        official_repo_path=official_repo_path,
        use_gpu_if_available=use_gpu,
    )
    manifest: dict[str, Any] = {
        "start_timestamp": utc_now_iso(),
        "end_timestamp": None,
        "task_names": task_names,
        "chunking_methods": methods,
        "max_chunk_tokens": max_chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "model_name": model_name,
        "seed": config.get("seed"),
        "project_git_commit": env_info.get("project_git_commit"),
        "official_repo_git_commit": env_info.get("official_repo_git_commit"),
        "device": env_info.get("device"),
        "gpu_name": env_info.get("gpu_name"),
        "package_versions": env_info.get("package_versions"),
        "numpy_stack_status": numpy_status,
        "dependency_workarounds": dependency_workarounds,
        "metrics": {},
        "notes": [
            "Indexing/retrieval split is not exposed separately by the official MTEB path; wall time is reported.",
        ],
        "output_file_paths": {},
    }
    manifest_path = boundary_dir / "boundary_chunking_run_manifest.json"
    write_json(manifest_path, manifest)

    task_module = importlib.import_module("chunked_pooling.chunked_eval_tasks")
    wrappers = importlib.import_module("chunked_pooling.wrappers")
    model, has_instructions = wrappers.load_model(model_name, None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                model = model.cuda()
        except Exception:
            pass
    model.eval()

    rows: list[dict[str, Any]] = []
    with patch_official_boundary_chunker(
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    ):
        for task_name in task_names:
            task_cls = getattr(task_module, task_name)
            corpus_stats_by_method: dict[str, dict[str, float]] = {}

            for method in methods:
                for pooling_label, late_enabled in [
                    ("traditional_chunking", False),
                    ("late_chunking", True),
                ]:
                    chunking_args = {
                        "chunk_size": max_chunk_tokens,
                        "n_sentences": 5,
                        "chunking_strategy": method,
                        "model_has_instructions": has_instructions,
                        "embedding_model_name": model_name,
                    }
                    task = task_cls(
                        chunked_pooling_enabled=late_enabled,
                        tokenizer=tokenizer,
                        prune_size=None,
                        truncate_max_length=None,
                        long_late_chunking_embed_size=0,
                        long_late_chunking_overlap_size=max_chunk_tokens,
                        **chunking_args,
                    )
                    evaluation = MTEB(
                        tasks=[task],
                        chunked_pooling_enabled=late_enabled,
                        tokenizer=tokenizer,
                        prune_size=None,
                        **chunking_args,
                    )
                    result_dir = boundary_dir / "results" / task_name / method / pooling_label
                    start = time.perf_counter()
                    previous_cwd = Path.cwd()
                    os.chdir(official_repo_path)
                    try:
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
                    wall_time = time.perf_counter() - start

                    if method not in corpus_stats_by_method:
                        corpus = _plain_split(task.corpus)
                        corpus_stats_by_method[method] = _corpus_chunk_stats(
                            corpus,
                            tokenizer,
                            method=method,
                            max_chunk_tokens=max_chunk_tokens,
                            overlap_tokens=overlap_tokens,
                            embedding_dim=embedding_dim,
                        )

                    metrics = _load_result_metrics(_find_task_result_json(result_dir, task_name))
                    row = {
                        "task": task_name,
                        "chunking_method": method,
                        "pooling": pooling_label,
                        "wall_time_seconds": wall_time,
                        "indexing_time_seconds": None,
                        "retrieval_time_seconds": None,
                        "timing_note": "official_mteb_combined_wall_time_only",
                        **corpus_stats_by_method[method],
                        **metrics,
                    }
                    rows.append(row)
                    manifest["metrics"].setdefault(task_name, {}).setdefault(method, {})[pooling_label] = row

    summary_path = boundary_dir / "tables" / "boundary_chunking_results_summary.csv"
    _write_csv(summary_path, rows)
    manifest["output_file_paths"]["boundary_results_summary_csv"] = str(summary_path)
    manifest["end_timestamp"] = utc_now_iso()
    write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    run_boundary_chunking_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
