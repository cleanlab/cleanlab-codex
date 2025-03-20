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


class IsBadResponseConfig(BaseModel):
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


DEFAULT_IS_BAD_RESPONSE_CONFIG: IsBadResponseConfig = IsBadResponseConfig(
    trustworthiness=0.5,
    response_helpfulness=0.5,
)


DEFAULT_TRUSTWORTHYRAG_CONFIG = {
    "options": {
        "log": ["explanation"],
    },
}


def get_default_trustworthyrag_config() -> dict[str, Any]:
    """Get the default configuration for the TrustworthyRAG."""
    return DEFAULT_TRUSTWORTHYRAG_CONFIG


def is_bad_response(scores: TrustworthyRAGScore, is_bad_response_config: IsBadResponseConfig | None = None) -> bool:
    """
    Check if the response is bad based on the scores computed by TrustworthyRAG and the config containing thresholds.
    """
    is_bad_response_config_dict: dict[str, float] = IsBadResponseConfig.model_validate(
        is_bad_response_config or DEFAULT_IS_BAD_RESPONSE_CONFIG
    ).model_dump()
    for eval_metric in BAD_RESPONSE_EVAL_METRICS:
        score = scores[eval_metric]["score"]
        if score is None:
            error_msg = f"Score for {eval_metric} is None"
            raise ValueError(error_msg)
        threshold = is_bad_response_config_dict[eval_metric]
        if score < threshold:
            return True
    return False
