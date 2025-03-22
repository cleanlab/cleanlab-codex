from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, field_validator

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


"""Evaluation metrics (excluding trustworthiness) that are used to determine if a response is bad."""
DEFAULT_EVAL_METRICS = ["response_helpfulness"]


class BadResponseThresholds(BaseModel):
    """Config for determining if a response is bad.
    Each key is an evaluation metric and the value is a threshold such that if the score is below the threshold, the response is bad.
    """

    trustworthiness: float = Field(
        description="Threshold for trustworthiness.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    response_helpfulness: float = Field(
        description="Threshold for response helpfulness.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    @property
    def default_threshold(self) -> float:
        """The default threshold to use when a specific evaluation metric's threshold is not set. This threshold is set to 0.5."""
        return 0.5

    def get_threshold(self, eval_name: str) -> float:
        """Get threshold for an eval if it exists.

        For fields defined in the model, returns their value (which may be the field's default).
        For custom evals not defined in the model, returns the default threshold value (see `default_threshold`).
        """

        # For fields defined in the model, use their value (which may be the field's default)
        if eval_name in self.model_fields:
            return cast(float, getattr(self, eval_name))

        # For custom evals, use the default threshold
        return getattr(self, eval_name, self.default_threshold)

    @field_validator("*")
    @classmethod
    def validate_threshold(cls, v: Any) -> float:
        """Validate that all fields (including dynamic ones) are floats between 0 and 1."""
        if not isinstance(v, (int, float)):
            error_msg = f"Threshold must be a number, got {type(v)}"
            raise TypeError(error_msg)
        if not 0 <= float(v) <= 1:
            error_msg = f"Threshold must be between 0 and 1, got {v}"
            raise ValueError(error_msg)
        return float(v)

    model_config = {
        "extra": "allow"  # Allow additional fields for custom eval thresholds
    }


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


def is_bad_response(
    scores: TrustworthyRAGScore | ThresholdedTrustworthyRAGScore, thresholds: BadResponseThresholds
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
