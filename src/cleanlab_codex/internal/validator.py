from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cleanlab_codex.utils.errors import MissingDependencyError

try:
    from cleanlab_tlm.utils.rag import Eval, TrustworthyRAGScore, get_default_evals
except ImportError as e:
    raise MissingDependencyError(
        import_name=e.name or "cleanlab-tlm",
        package_url="https://github.com/cleanlab/cleanlab-tlm",
    ) from e


"""Evaluation metrics (excluding trustworthiness) that are used to determine if a response is bad."""
EVAL_METRICS = ["response_helpfulness"]

"""Evaluation metrics that are used to determine if a response is bad."""
BAD_RESPONSE_EVAL_METRICS = ["trustworthiness", *EVAL_METRICS]



class BadResponseThresholds(BaseModel):
    """Config for determining if a response is bad.
    Each key is an evaluation metric and the value is a threshold such that if the score is below the threshold, the response is bad.
    """

    trustworthiness: float = Field(
        description="Threshold for trustworthiness. If the score is below this threshold, the response is bad.",
        default=0.5,
        ge=0,
        le=1,
    )
    response_helpfulness: float = Field(
        description="Threshold for response helpfulness. If the score is below this threshold, the response is bad.",
        default=0.5,
        ge=0,
        le=1,
    )


def get_default_evaluations() -> list[Eval]:
    """Get the default evaluations for the TrustworthyRAG.

    Note:
        This excludes trustworthiness, which is automatically computed by TrustworthyRAG.
    """
    return [evaluation for evaluation in get_default_evals() if evaluation.name in EVAL_METRICS]


DEFAULT_TRUSTWORTHYRAG_CONFIG = {
    "options": {
        "log": ["explanation"],
    },
}


def get_default_trustworthyrag_config() -> dict[str, Any]:
    """Get the default configuration for the TrustworthyRAG."""
    return DEFAULT_TRUSTWORTHYRAG_CONFIG


def is_bad_response(scores: TrustworthyRAGScore, thresholds: dict[str, float]) -> bool:
    """
    Check if the response is bad based on the scores computed by TrustworthyRAG and the config containing thresholds.
    """
    for eval_metric, threshold in thresholds.items():
        if scores[eval_metric]["score"] < threshold:
            return True
    return False
