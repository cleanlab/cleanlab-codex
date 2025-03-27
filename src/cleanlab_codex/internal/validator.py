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
