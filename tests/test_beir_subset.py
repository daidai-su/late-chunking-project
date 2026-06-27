from latechunk_project.beir_subset import (
    filter_qrels_by_query_ids,
    make_beir_smoke_subset,
    sample_corpus_preserving_relevance,
    sample_query_ids,
)


def test_filter_qrels_by_selected_query_ids():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}

    assert filter_qrels_by_query_ids(qrels, {"q2"}) == {"q2": {"d2": 1}}


def test_sample_corpus_without_removing_relevant_docs():
    corpus = {
        "d1": {"title": "A", "text": "alpha"},
        "d2": {"title": "B", "text": "beta"},
        "d3": {"title": "C", "text": "gamma"},
    }
    qrels = {"q1": {"d1": 1, "d2": 1}}

    subset = sample_corpus_preserving_relevance(corpus, qrels, max_docs=1, seed=42)

    assert set(subset) == {"d1", "d2"}


def test_deterministic_sampling_with_seed():
    queries = {f"q{i}": f"query {i}" for i in range(10)}

    first = sample_query_ids(queries, max_queries=4, seed=123)
    second = sample_query_ids(queries, max_queries=4, seed=123)
    different = sample_query_ids(queries, max_queries=4, seed=456)

    assert first == second
    assert first != different


def test_make_beir_smoke_subset_keeps_consistent_ids():
    corpus = {f"d{i}": {"title": "", "text": str(i)} for i in range(6)}
    queries = {"q1": "one", "q2": "two"}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}

    subset_corpus, subset_queries, subset_qrels = make_beir_smoke_subset(
        corpus,
        queries,
        qrels,
        max_queries=1,
        max_corpus_docs=2,
        seed=7,
    )

    assert set(subset_queries) == set(subset_qrels)
    kept_relevant_docs = {doc_id for scores in subset_qrels.values() for doc_id in scores}
    assert kept_relevant_docs <= set(subset_corpus)

