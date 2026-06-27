import json
from tempfile import TemporaryDirectory
from pathlib import Path

from latechunk_project.result_parsing import (
    method_name_from_result_dir,
    parse_mteb_result_json,
    parse_official_metrics,
    summarize_mteb_result_jsons,
)


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


def test_parse_mteb_result_json():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "SciFactChunked.json"
        path.write_text(
            json.dumps(
                {
                    "dataset_revision": "abc",
                    "evaluation_time": 12.5,
                    "mteb_version": "1.0",
                    "scores": {
                        "test": [
                            {
                                "main_score": 0.66,
                                "ndcg_at_10": 0.66,
                                "recall_at_10": 0.77,
                                "map_at_10": 0.61,
                                "mrr_at_10": 0.63,
                            }
                        ]
                    },
                    "task_name": "SciFactChunked",
                }
            ),
            encoding="utf-8",
        )

        parsed = parse_mteb_result_json(path)

        assert parsed["task_name"] == "SciFactChunked"
        assert parsed["metrics"]["nDCG@10"] == 0.66
        assert parsed["metrics"]["Recall@10"] == 0.77


def test_summarize_mteb_result_jsons():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        result_dir = root / "results-chunked-pooling" / "no_model" / "no_rev"
        result_dir.mkdir(parents=True)
        result_path = result_dir / "SciFactChunked.json"
        result_path.write_text(
            json.dumps(
                {
                    "evaluation_time": 46.2,
                    "scores": {
                        "test": [
                            {
                                "main_score": 0.66098,
                                "ndcg_at_10": 0.66098,
                                "recall_at_10": 0.77756,
                                "map_at_10": 0.61829,
                                "mrr_at_10": 0.63372,
                            }
                        ]
                    },
                    "task_name": "SciFactChunked",
                }
            ),
            encoding="utf-8",
        )

        rows, manifest_metrics = summarize_mteb_result_jsons(root, ["SciFactChunked"])

        assert method_name_from_result_dir(result_path) == "chunked_pooling"
        assert rows[0]["method"] == "chunked_pooling"
        assert rows[0]["ndcg_at_10"] == 0.66098
        assert manifest_metrics["SciFactChunked"]["chunked_pooling"]["recall_at_10"] == 0.77756
