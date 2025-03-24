"""
Leverage Cleanlab's Evals and Codex to detect and remediate bad responses in RAG applications.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from pydantic import BaseModel, Field, field_validator

from cleanlab_codex.internal.validator import (
    get_default_evaluations,
    get_default_trustworthyrag_config,
)
from cleanlab_codex.internal.validator import is_bad_response as _is_bad_response
from cleanlab_codex.internal.validator import update_scores_based_on_thresholds as _update_scores_based_on_thresholds
from cleanlab_codex.project import Project
from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore
from cleanlab_codex.utils.errors import MissingDependencyError

try:
    from cleanlab_tlm import TrustworthyRAG
except ImportError as e:
    raise MissingDependencyError(
        import_name=e.name or "cleanlab-tlm",
        package_url="https://github.com/cleanlab/cleanlab-tlm",
    ) from e


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


class Validator:
    def __init__(
        self,
        codex_access_key: str,
        tlm_api_key: Optional[str] = None,
        trustworthy_rag_config: Optional[dict[str, Any]] = None,
        bad_response_thresholds: Optional[dict[str, float]] = None,
    ):
        """Evaluates the quality of responses generated in RAG applications and remediates them if needed.

        This object combines Cleanlab's various Evals with thresholding to detect bad responses and remediates them with Codex.

        Args:
            codex_access_key (str): The [access key](/codex/web_tutorials/create_project/#access-keys) for a Codex project.
            tlm_api_key (Optional[str]): The API key for [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag).
            trustworthy_rag_config (Optional[dict[str, Any]]): Optional initialization arguments for [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag), which is used to detect response issues.
            bad_response_thresholds (Optional[dict[str, float]]): Detection score thresholds used to flag whether or not a response is considered bad. Each key in this dict corresponds to an Eval from TrustworthyRAG, and the value indicates a threshold below which scores from this Eval are considered detected issues.  A response is flagged as bad if any issues are detected for it.
        """
        trustworthy_rag_config = trustworthy_rag_config or get_default_trustworthyrag_config()
        if tlm_api_key is not None and "api_key" in trustworthy_rag_config:
            error_msg = "Cannot specify both tlm_api_key and api_key in trustworthy_rag_config"
            raise ValueError(error_msg)
        if tlm_api_key is not None:
            trustworthy_rag_config["api_key"] = tlm_api_key

        self._project: Project = Project.from_access_key(access_key=codex_access_key)

        trustworthy_rag_config.setdefault("evals", get_default_evaluations())
        self._tlm_rag = TrustworthyRAG(**trustworthy_rag_config)

        # Validate that all the necessary thresholds are present in the TrustworthyRAG.
        _evals = [e.name for e in self._tlm_rag.get_evals()] + ["trustworthiness"]

        self._bad_response_thresholds = BadResponseThresholds.model_validate(bad_response_thresholds or {})

        _threshold_keys = self._bad_response_thresholds.model_dump().keys()

        # Check if there are any thresholds without corresponding evals (this is an error)
        _extra_thresholds = set(_threshold_keys) - set(_evals)
        if _extra_thresholds:
            error_msg = f"Found thresholds for non-existent evaluation metrics: {_extra_thresholds}"
            raise ValueError(error_msg)

    def validate(
        self,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> dict[str, Any]:
        """Evaluate whether the AI-generated response is bad, and if so, request an alternate expert response.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'is_bad_response': True if the response is flagged as potentially bad, False otherwise.
                - 'expert_answer': Alternate SME-provided answer from Codex, or None if no answer could be found in the Codex Project.
                - Additional keys: Various keys from a [`ThresholdedTrustworthyRAGScore`](/cleanlab_codex/types/validator/#class-thresholdedtrustworthyragscore) dictionary, with raw scores from [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag) for each evaluation metric.  `is_bad` indicating whether the score is below the threshold.
        """
        scores, is_bad_response = self.detect(query, context, response, prompt, form_prompt)
        expert_answer = None
        if is_bad_response:
            expert_answer = self.remediate(query)

        return {
            "is_bad_response": is_bad_response,
            "expert_answer": expert_answer,
            **scores,
        }

    def detect(
        self,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> tuple[ThresholdedTrustworthyRAGScore, bool]:
        """Evaluate the response quality using TrustworthyRAG and determine if it is a bad response.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.

        Returns:
            tuple[ThresholdedTrustworthyRAGScore, bool]: A tuple containing:
                - ThresholdedTrustworthyRAGScore: Quality scores for different evaluation metrics like trustworthiness
                  and response helpfulness. Each metric has a score between 0-1. It also has a boolean flag, `is_bad` indicating whether the score is below a given threshold.
                - bool: True if the response is determined to be bad based on the evaluation scores
                  and configured thresholds, False otherwise.
        """
        scores = cast(
            ThresholdedTrustworthyRAGScore,
            self._tlm_rag.score(
                response=response,
                query=query,
                context=context,
                prompt=prompt,
                form_prompt=form_prompt,
            ),
        )

        _update_scores_based_on_thresholds(scores, thresholds=self._bad_response_thresholds)

        is_bad_response = _is_bad_response(scores, self._bad_response_thresholds)
        return scores, is_bad_response

    def remediate(self, query: str) -> str | None:
        """Request a SME-provided answer for this query, if one is available in Codex.

        Args:
            query (str): The user's original query to get SME-provided answer for.

        Returns:
            str | None: The SME-provided answer from Codex, or None if no answer could be found in the Codex Project.
        """
        codex_answer, _ = self._project.query(question=query)
        return codex_answer
