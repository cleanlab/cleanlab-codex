from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cleanlab_codex.utils.errors import MissingDependencyError

try:
    from cleanlab_tlm.utils.rag import Eval, TrustworthyRAGScore, get_default_evals
except ImportError as e:
    raise MissingDependencyError(
        import_name=e.name or "cleanlab-tlm",
        package_url="https://github.com/cleanlab/cleanlab-tlm",
    ) from e

if TYPE_CHECKING:
    from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore
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
    scores: ThresholdedTrustworthyRAGScore, thresholds: BadResponseThresholds
) -> None:
    """Adds a `is_bad` flag to the scores dictionaries based on the thresholds."""
    for eval_name, score_dict in scores.items():
        score_dict.setdefault("is_bad", False)
        if (score := score_dict["score"]) is not None:
            score_dict["is_bad"] = score < thresholds.get_threshold(eval_name)


def is_bad_response(
    scores: TrustworthyRAGScore | ThresholdedTrustworthyRAGScore,
    thresholds: BadResponseThresholds,
) -> bool:
    """
    Check if the response is bad based on the scores computed by TrustworthyRAG and the config containing thresholds.
    """
    for eval_metric, score_dict in scores.items():
        score = score_dict["score"]
        if score is None:
            continue
        if score < thresholds.get_threshold(eval_metric):
            return True
    return False
