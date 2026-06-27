from latechunk_project.bm25_fusion import bm25_rankings, rrf_fuse_rankings, simple_tokenize


def test_bm25_ranking_works_on_toy_corpus():
    corpus = {
        "d1": {"text": "alpha beta"},
        "d2": {"text": "gamma delta"},
    }
    queries = {"q1": "alpha"}

    rankings, diagnostics = bm25_rankings(corpus, queries)

    assert rankings["q1"][0]["doc_id"] == "d1"
    assert diagnostics["num_corpus_docs"] == 2


def test_rrf_fusion_works_on_toy_rankings():
    dense = {"q1": [{"doc_id": "d1", "doc_rank": 1}, {"doc_id": "d2", "doc_rank": 2}]}
    bm25 = {"q1": [{"doc_id": "d2", "doc_rank": 1}, {"doc_id": "d3", "doc_rank": 2}]}

    fused = rrf_fuse_rankings(dense, bm25, rrf_k=60)

    assert fused["q1"][0]["doc_id"] == "d2"
    assert {row["doc_id"] for row in fused["q1"]} == {"d1", "d2", "d3"}


def test_rrf_fusion_handles_missing_docs_safely():
    fused = rrf_fuse_rankings({"q1": [{"doc_id": "d1", "doc_rank": 1}]}, {}, rrf_k=60)

    assert fused["q1"][0]["doc_id"] == "d1"


def test_rrf_fusion_deterministic_ordering():
    dense = {"q1": [{"doc_id": "b", "doc_rank": 1}, {"doc_id": "a", "doc_rank": 1}]}
    fused = rrf_fuse_rankings(dense, {}, rrf_k=60)

    assert [row["doc_id"] for row in fused["q1"]] == ["a", "b"]


def test_simple_tokenize_is_deterministic():
    assert simple_tokenize("Alpha, beta! Alpha") == ["alpha", "beta", "alpha"]

