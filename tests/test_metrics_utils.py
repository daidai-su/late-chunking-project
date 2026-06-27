import math

from latechunk_project.metrics_utils import compute_ndcg_at_k, compute_recall_at_k


def test_compute_ndcg_at_10_on_toy_ranking():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    rankings = {"q1": ["d1", "d2", "d3"]}

    assert math.isclose(compute_ndcg_at_k(rankings, qrels, k=10), 1.0)


def test_compute_recall_at_k_on_toy_ranking():
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}
    rankings = {"q1": ["d1", "d3", "d2"], "q2": ["d4", "d3"]}

    assert math.isclose(compute_recall_at_k(rankings, qrels, k=2), 0.75)


def test_empty_rankings_are_safe():
    assert compute_ndcg_at_k({}, {}, k=10) == 0.0
    assert compute_recall_at_k({}, {}, k=10) == 0.0
    assert compute_ndcg_at_k({"q1": []}, {"q1": {"d1": 1}}, k=10) == 0.0
    assert compute_recall_at_k({"q1": []}, {"q1": {"d1": 1}}, k=10) == 0.0
