from latechunk_project.result_parsing import parse_official_metrics


def test_parse_toy_official_output_containing_ndcg_at_10():
    output = """
    Running task SciFactChunked
    nDCG@10: 0.4567
    Recall@100 = 0.891
    """

    assert parse_official_metrics(output)["nDCG@10"] == 0.4567
    assert parse_official_metrics(output)["Recall@100"] == 0.891


def test_parse_dictionary_style_metric_names():
    output = "{'ndcg_at_10': 0.25, 'recall_at_10': 0.5}"

    metrics = parse_official_metrics(output)

    assert metrics == {"nDCG@10": 0.25, "Recall@10": 0.5}


def test_missing_metrics_returns_empty_dict():
    assert parse_official_metrics("finished without printing final scores") == {}

