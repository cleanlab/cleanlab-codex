from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

from cleanlab_tlm.utils.rag import Eval, TrustworthyRAGScore, get_default_evals

from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore

if TYPE_CHECKING:
    from cleanlab_codex.validator import BadResponseThresholds


"""Evaluation metrics (excluding trustworthiness) that are used to determine if a response is bad."""
DEFAULT_EVAL_METRICS = ["response_helpfulness"]


def get_default_evaluations() -> list[Eval]:
    """Get the default evaluations for the TrustworthyRAG.

    Note:
        This excludes trustworthiness, which is automatically computed by TrustworthyRAG.
    """
    return [evaluation for evaluation in get_default_evals() if evaluation.name in DEFAULT_EVAL_METRICS]


def get_default_trustworthyrag_config() -> dict[str, Any]:
    """Get the default configuration for the TrustworthyRAG."""
    return {
        "options": {
            "log": ["explanation"],
        },
    }


def update_scores_based_on_thresholds(
    scores: TrustworthyRAGScore | Sequence[TrustworthyRAGScore], thresholds: BadResponseThresholds
) -> ThresholdedTrustworthyRAGScore:
    """Adds a `is_bad` flag to the scores dictionaries based on the thresholds."""

    # Helper function to check if a score is bad
    def is_bad(score: Optional[float], threshold: float) -> bool:
        return score is not None and score < threshold

    if isinstance(scores, Sequence):
        raise NotImplementedError("Batching is not supported yet.")

    thresholded_scores = {}
    for eval_name, score_dict in scores.items():
        thresholded_scores[eval_name] = {
            **score_dict,
            "is_bad": is_bad(score_dict["score"], thresholds.get_threshold(eval_name)),
        }
    return cast(ThresholdedTrustworthyRAGScore, thresholded_scores)


def process_score_metadata(scores: ThresholdedTrustworthyRAGScore, thresholds: BadResponseThresholds) -> dict[str, Any]:
    """Process scores into metadata format with standardized keys.

    Args:
        scores: The ThresholdedTrustworthyRAGScore containing evaluation results
        thresholds: The BadResponseThresholds configuration

    Returns:
        dict: A dictionary containing evaluation scores and their corresponding metadata
    """
    metadata: dict[str, Any] = {}


    # Simple mappings for is_bad keys
    score_to_is_bad_key = {
        "trustworthiness": "is_not_trustworthy",
        "query_ease": "is_not_query_easy",
        "response_helpfulness": "is_not_response_helpful",
        "context_sufficiency": "is_context_insufficient",
    }

    # Process scores and add to metadata
    for metric, score_data in scores.items():
        metadata[metric] = score_data["score"]

        # Add is_bad flags with standardized naming
        is_bad_key = score_to_is_bad_key.get(metric, f"is_not_{metric}")
        metadata[is_bad_key] = score_data["is_bad"]

        # Special case for trustworthiness explanation
        if metric == "trustworthiness" and "log" in score_data and "explanation" in score_data["log"]:
            metadata["explanation"] = score_data["log"]["explanation"]

    # Add thresholds to metadata
    metadata["thresholds"] = thresholds.model_dump()

    return metadata
