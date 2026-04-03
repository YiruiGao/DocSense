import pytest

from app.evaluation.metrics.retrieval import calculate_hit_rate, calculate_mrr


@pytest.mark.unit
def test_calculate_hit_rate_uses_first_matching_rank():
    results = [
        {
            "method": "hybrid",
            "retrieved_chunk_ids": ["chunk-b", "chunk-a", "chunk-c"],
            "expected_chunk_ids": ["chunk-a"],
        },
        {
            "method": "hybrid",
            "retrieved_chunk_ids": ["chunk-d", "chunk-e"],
            "expected_chunk_ids": ["chunk-z"],
        },
    ]

    metric = calculate_hit_rate(results)

    assert metric.method_name == "hybrid"
    assert metric.total_queries == 2
    assert metric.questions_with_hits == 1
    assert metric.hit_at_3 == 0.5
    assert metric.hit_at_5 == 0.5
    assert metric.hit_at_10 == 0.5


@pytest.mark.unit
def test_calculate_mrr_counts_missing_hits_as_zero():
    results = [
        {"rank": 1, "method": "baseline"},
        {"rank": 4, "method": "baseline"},
        {"rank": None, "method": "baseline"},
    ]

    metric = calculate_mrr(results)

    assert metric.method_name == "baseline"
    assert metric.total_queries == 3
    assert metric.reciprocal_ranks == [1.0, 0.25, 0.0]
    assert metric.mrr == pytest.approx((1.0 + 0.25) / 3)
