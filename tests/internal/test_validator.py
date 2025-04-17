from typing import cast

from cleanlab_tlm.utils.rag import TrustworthyRAGScore

from cleanlab_codex.internal.validator import (
    get_default_evaluations,
    process_score_metadata,
    update_scores_based_on_thresholds,
)
from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore
from cleanlab_codex.validator import BadResponseThresholds


def make_scores(trustworthiness: float, response_helpfulness: float) -> TrustworthyRAGScore:
    scores = {
        "trustworthiness": {
            "score": trustworthiness,
        },
        "response_helpfulness": {
            "score": response_helpfulness,
        },
    }
    return cast(TrustworthyRAGScore, scores)


def make_is_bad_response_config(trustworthiness: float, response_helpfulness: float) -> BadResponseThresholds:
    return BadResponseThresholds(
        trustworthiness=trustworthiness,
        response_helpfulness=response_helpfulness,
    )


def test_get_default_evaluations() -> None:
    assert {evaluation.name for evaluation in get_default_evaluations()} == {"response_helpfulness"}


def test_process_score_metadata() -> None:
    # Create test scores with various metrics
    thresholded_scores = {
        "trustworthiness": {"score": 0.8, "is_bad": False, "log": {"explanation": "Test explanation"}},
        "response_helpfulness": {"score": 0.6, "is_bad": True},
        "query_ease": {"score": 0.9, "is_bad": False},
    }

    thresholds = BadResponseThresholds(trustworthiness=0.7, response_helpfulness=0.7)

    metadata = process_score_metadata(cast(ThresholdedTrustworthyRAGScore, thresholded_scores), thresholds)

    # Check scores and flags
    expected_metadata = {
        "trustworthiness": 0.8,
        "response_helpfulness": 0.6,
        "query_ease": 0.9,
        "is_not_trustworthy": False,
        "is_not_response_helpful": True,
        "is_not_query_easy": False,
        "explanation": "Test explanation",
        "thresholds": {"trustworthiness": 0.7, "response_helpfulness": 0.7, "query_ease": 0.0},
        "label": "unhelpful",
    }

    assert metadata == expected_metadata


def test_process_score_metadata_edge_cases() -> None:
    """Test edge cases for process_score_metadata."""
    thresholds = BadResponseThresholds()

    # Test empty scores
    metadata_for_empty_scores = process_score_metadata(cast(ThresholdedTrustworthyRAGScore, {}), thresholds)
    assert {"thresholds", "label"} == set(metadata_for_empty_scores.keys())

    # Test missing explanation
    scores = cast(ThresholdedTrustworthyRAGScore, {"trustworthiness": {"score": 0.6, "is_bad": True}})
    metadata_missing_explanation = process_score_metadata(scores, thresholds)
    assert "explanation" not in metadata_missing_explanation

    # Test custom metric
    scores = cast(ThresholdedTrustworthyRAGScore, {"my_metric": {"score": 0.3, "is_bad": True}})
    metadata_custom_metric = process_score_metadata(scores, thresholds)
    assert metadata_custom_metric["my_metric"] == 0.3
    assert metadata_custom_metric["is_not_my_metric"] is True


def test_update_scores_based_on_thresholds() -> None:
    """Test that update_scores_based_on_thresholds correctly flags scores based on thresholds."""
    raw_scores = cast(
        TrustworthyRAGScore,
        {
            "trustworthiness": {"score": 0.6},  # Below threshold
            "response_helpfulness": {"score": 0.8},  # Above threshold
            "custom_metric": {"score": 0.4},  # Below custom threshold
            "another_metric": {"score": 0.6},  # Uses default threshold
        },
    )

    thresholds = BadResponseThresholds(trustworthiness=0.7, response_helpfulness=0.7, custom_metric=0.45)  # type: ignore[call-arg]

    scores = update_scores_based_on_thresholds(raw_scores, thresholds)

    expected_is_bad = {
        "trustworthiness": True,
        "response_helpfulness": False,
        "custom_metric": True,
        "another_metric": False,
    }

    for metric, expected in expected_is_bad.items():
        assert scores[metric]["is_bad"] is expected
    assert all(scores[k]["score"] == raw_scores[k]["score"] for k in raw_scores)
