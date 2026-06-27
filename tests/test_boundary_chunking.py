from latechunk_project.boundary_chunking import (
    ChunkSpan,
    _ensure_task_k_values,
    boundary_chunk_spans,
    chunk_text_by_whitespace,
    overlap_token_spans,
    pack_units_to_max_tokens,
    sentence_unit_spans,
    validate_non_overlapping_coverage,
    validate_overlapping_coverage,
    whitespace_token_offsets,
)


def _span_lengths(spans):
    return [span.length for span in spans]


def test_punctuation_sentence_splitter():
    text = "Alpha beta. Gamma? Delta!"
    offsets = whitespace_token_offsets(text)

    spans = sentence_unit_spans(text, offsets)

    assert spans == [ChunkSpan(0, 2), ChunkSpan(2, 3), ChunkSpan(3, 4)]


def test_max_token_packing_splits_long_units():
    units = [ChunkSpan(0, 3), ChunkSpan(3, 9), ChunkSpan(9, 11)]

    spans = pack_units_to_max_tokens(units, max_chunk_tokens=4)

    assert spans == [ChunkSpan(0, 3), ChunkSpan(3, 7), ChunkSpan(7, 9), ChunkSpan(9, 11)]
    assert max(_span_lengths(spans)) <= 4


def test_boundary_methods_do_not_emit_empty_chunks():
    text = "One short sentence.\n\nSecond paragraph has a few words. Third sentence."
    offsets = whitespace_token_offsets(text)

    for method in ["fixed_256_tokens", "sentence_boundary_approx", "paragraph_boundary_approx"]:
        spans = boundary_chunk_spans(text, offsets, method=method, max_chunk_tokens=4)
        assert spans
        assert all(span.length > 0 for span in spans)


def test_chunk_spans_cover_all_tokens_in_order():
    text = "One two three. Four five six seven. Eight nine ten."
    offsets = whitespace_token_offsets(text)

    for method in ["fixed_256_tokens", "sentence_boundary_approx", "paragraph_boundary_approx"]:
        spans = boundary_chunk_spans(text, offsets, method=method, max_chunk_tokens=4)
        assert validate_non_overlapping_coverage(spans, len(offsets))


def test_overlap_chunking_deterministic():
    first = overlap_token_spans(num_tokens=20, max_chunk_tokens=8, overlap_tokens=3)
    second = overlap_token_spans(num_tokens=20, max_chunk_tokens=8, overlap_tokens=3)

    assert first == second
    assert first == [ChunkSpan(0, 8), ChunkSpan(5, 13), ChunkSpan(10, 18), ChunkSpan(15, 20)]
    assert validate_overlapping_coverage(first, 20)


def test_chunk_text_by_whitespace_overlap_uses_fixed_settings():
    text = " ".join(f"tok{i}" for i in range(12))

    spans = chunk_text_by_whitespace(
        text,
        method="overlap_fixed",
        max_chunk_tokens=5,
        overlap_tokens=2,
    )

    assert spans == [ChunkSpan(0, 5), ChunkSpan(3, 8), ChunkSpan(6, 11), ChunkSpan(9, 12)]


def test_task_k_values_include_recall_100():
    class Task:
        k_values = [1, 3, 10]

    values = _ensure_task_k_values(Task(), required_k_values=(10, 100))

    assert values == [1, 3, 10, 100]
