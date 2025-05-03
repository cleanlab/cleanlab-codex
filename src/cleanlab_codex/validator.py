"""
Detect and remediate bad responses in RAG applications, by integrating Codex as-a-Backup.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from cleanlab_tlm import TrustworthyRAG
from pydantic import BaseModel, Field, field_validator

from cleanlab_codex.internal.validator import (
    get_default_evaluations,
    get_default_trustworthyrag_config,
)
from cleanlab_codex.internal.validator import (
    process_score_metadata as _process_score_metadata,
)
from cleanlab_codex.internal.validator import (
    update_scores_based_on_thresholds as _update_scores_based_on_thresholds,
)
from cleanlab_codex.project import Project

if TYPE_CHECKING:
    from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore


class Validator:
    def __init__(
        self,
        codex_access_key: str,
        tlm_api_key: Optional[str] = None,
        trustworthy_rag_config: Optional[dict[str, Any]] = None,
        bad_response_thresholds: Optional[dict[str, float]] = None,
    ):
        """Real-time detection and remediation of bad responses in RAG applications, powered by Cleanlab's TrustworthyRAG and Codex.

        This object combines Cleanlab's TrustworthyRAG evaluation scores with configurable thresholds to detect potentially bad responses
        in your RAG application. When a bad response is detected, this Validator automatically attempts to remediate by retrieving an expert-provided
        answer from the Codex Project you've integrated with your RAG app. If no expert answer is available,
        the corresponding query is logged in the Codex Project for SMEs to answer.

        For production, use the `validate()` method which provides a complete validation workflow including both detection and remediation.
        A `detect()` method is separately available for you to test/tune detection configurations like score thresholds and TrustworthyRAG settings
        without triggering any Codex lookups that otherwise could affect the state of the corresponding Codex Project.

        Args:
            codex_access_key (str): The [access key](/codex/web_tutorials/create_project/#access-keys) for a Codex project. Used to retrieve expert-provided answers
                when bad responses are detected, or otherwise log the corresponding queries for SMEs to answer.

            tlm_api_key (str, optional): API key for accessing [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag). If not provided, this must be specified
                in `trustworthy_rag_config`.

            trustworthy_rag_config (dict[str, Any], optional): Optional initialization arguments for [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag),
                which is used to detect response issues. If not provided, a default configuration will be used.
                By default, this Validator uses the same default configurations as [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag), except:
                - Explanations are returned in logs for better debugging
                - Only the `response_helpfulness` eval is run

            bad_response_thresholds (dict[str, float], optional): Detection score thresholds used to flag whether
                a response is bad or not. Each key corresponds to an Eval from [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag),
                and the value indicates a threshold (between 0 and 1) below which Eval scores are treated as detected issues. A response
                is flagged as bad if any issues are detected. If not provided or only partially provided, default thresholds will be used
                for any missing metrics. Note that if a threshold is provided for a metric, that metric must correspond to an eval
                that is configured to run (with the exception of 'trustworthiness' which is always implicitly configured). You can
                configure arbitrary evals to run, and their thresholds will use default values unless explicitly set. See
                [`BadResponseThresholds`](/codex/api/python/validator/#class-badresponsethresholds) for more details on the default values.

        Raises:
            ValueError: If both tlm_api_key and api_key in trustworthy_rag_config are provided.
            ValueError: If any user-provided thresholds in bad_response_thresholds are for evaluation metrics that are not available in the current evaluation setup (except 'trustworthiness' which is always available).
            TypeError: If any threshold value is not a number.
            ValueError: If any threshold value is not between 0 and 1.
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

        # Only check thresholds that were explicitly provided by the user
        if bad_response_thresholds is not None:
            _threshold_keys = bad_response_thresholds.keys()
            # Check if there are any user-provided thresholds without corresponding evals
            _extra_thresholds = set(_threshold_keys) - set(_evals)
            if _extra_thresholds:
                error_msg = (
                    f"Found thresholds for metrics that are not available: {_extra_thresholds}. "
                    "Available metrics are determined by the evals parameter provided to TrustworthyRAG plus 'trustworthiness' which is always included."
                )
                raise ValueError(error_msg)

    def validate(
        self,
        *,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        log_results: bool = True,
    ) -> dict[str, Any]:
        """Evaluate whether the AI-generated response is bad, and if so, request an alternate expert answer.
        If no expert answer is available, this query is still logged for SMEs to answer.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.
            prompt (str, optional): Optional prompt representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
            form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other. This function should take query and context as parameters and return a formatted prompt string. If not provided, a default prompt formatter will be used. To include a system prompt or any other special instructions for your LLM, incorporate them directly in your custom form_prompt() function definition.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'expert_answer': Alternate SME-provided answer from Codex if the response was flagged as bad and an answer was found in the Codex Project, or None otherwise.
                - 'is_bad_response': True if the response is flagged as potentially bad, False otherwise. When True, a Codex lookup is performed, which logs this query into the Codex Project for SMEs to answer.
                - Additional keys from a [`ThresholdedTrustworthyRAGScore`](/codex/api/python/types.validator/#class-thresholdedtrustworthyragscore) dictionary: each corresponds to a [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag) evaluation metric, and points to the score for this evaluation as well as a boolean `is_bad` flagging whether the score falls below the corresponding threshold.
        """
        scores, is_bad_response = self.detect(
            query=query, context=context, response=response, prompt=prompt, form_prompt=form_prompt
        )
        expert_answer = None
        if is_bad_response:
            final_metadata = deepcopy(metadata) if metadata else {}
            if log_results:
                processed_metadata = _process_score_metadata(scores, self._bad_response_thresholds)
                final_metadata.update(processed_metadata)
            expert_answer = self._remediate(query=query, metadata=final_metadata)

        self._project.increment_queries()

        return {
            "expert_answer": expert_answer,
            "is_bad_response": is_bad_response,
            **scores,
        }

    async def validate_async(
        self,
        *,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        log_results: bool = True,
    ) -> dict[str, Any]:
        """Evaluate whether the AI-generated response is bad, and if so, request an alternate expert answer.
        If no expert answer is available, this query is still logged for SMEs to answer.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.
            prompt (str, optional): Optional prompt representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
            form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other. This function should take query and context as parameters and return a formatted prompt string. If not provided, a default prompt formatter will be used. To include a system prompt or any other special instructions for your LLM, incorporate them directly in your custom form_prompt() function definition.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'expert_answer': Alternate SME-provided answer from Codex if the response was flagged as bad and an answer was found in the Codex Project, or None otherwise.
                - 'is_bad_response': True if the response is flagged as potentially bad, False otherwise. When True, a Codex lookup is performed, which logs this query into the Codex Project for SMEs to answer.
                - Additional keys from a [`ThresholdedTrustworthyRAGScore`](/codex/api/python/types.validator/#class-thresholdedtrustworthyragscore) dictionary: each corresponds to a [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag) evaluation metric, and points to the score for this evaluation as well as a boolean `is_bad` flagging whether the score falls below the corresponding threshold.
        """
        scores, is_bad_response = await self.detect_async(query, context, response, prompt, form_prompt)
        final_metadata = metadata.copy() if metadata else {}
        if log_results:
            processed_metadata = _process_score_metadata(scores, self._bad_response_thresholds)
            final_metadata.update(processed_metadata)

        expert_answer = None
        if is_bad_response:
            expert_answer = self._remediate(query=query, metadata=final_metadata)

        self._project.increment_queries()

        return {
            "expert_answer": expert_answer,
            "is_bad_response": is_bad_response,
            **scores,
        }

    def detect(
        self,
        *,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> tuple[ThresholdedTrustworthyRAGScore, bool]:
        """Score response quality using TrustworthyRAG and flag bad responses based on configured thresholds.

        Note:
            Use this method instead of `validate()` to test/tune detection configurations like score thresholds and TrustworthyRAG settings.
            This `detect()` method will not affect your Codex Project, whereas `validate()` will log queries whose response was detected as bad into the Codex Project and is thus only suitable for production, not testing.
            Both this method and `validate()` rely on this same detection logic, so you can use this method to first optimize detections and then switch to using `validate()`.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.
            prompt (str, optional): Optional prompt representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
            form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other. This function should take query and context as parameters and return a formatted prompt string. If not provided, a default prompt formatter will be used. To include a system prompt or any other special instructions for your LLM, incorporate them directly in your custom form_prompt() function definition.

        Returns:
            tuple[ThresholdedTrustworthyRAGScore, bool]: A tuple containing:
                - ThresholdedTrustworthyRAGScore: Quality scores for different evaluation metrics like trustworthiness
                  and response helpfulness. Each metric has a score between 0-1. It also has a boolean flag, `is_bad` indicating whether the score is below a given threshold.
                - bool: True if the response is determined to be bad based on the evaluation scores
                  and configured thresholds, False otherwise.
        """
        scores = self._tlm_rag.score(
            response=response,
            query=query,
            context=context,
            prompt=prompt,
            form_prompt=form_prompt,
        )

        thresholded_scores = _update_scores_based_on_thresholds(
            scores=scores,
            thresholds=self._bad_response_thresholds,
        )

        is_bad_response = any(score_dict["is_bad"] for score_dict in thresholded_scores.values())
        return thresholded_scores, is_bad_response

    async def detect_async(
        self,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> tuple[ThresholdedTrustworthyRAGScore, bool]:
        """Score response quality using TrustworthyRAG and flag bad responses based on configured thresholds.

        Note:
            Use this method instead of `validate()` to test/tune detection configurations like score thresholds and TrustworthyRAG settings.
            This `detect()` method will not affect your Codex Project, whereas `validate()` will log queries whose response was detected as bad into the Codex Project and is thus only suitable for production, not testing.
            Both this method and `validate()` rely on this same detection logic, so you can use this method to first optimize detections and then switch to using `validate()`.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.
            prompt (str, optional): Optional prompt representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
            form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other. This function should take query and context as parameters and return a formatted prompt string. If not provided, a default prompt formatter will be used. To include a system prompt or any other special instructions for your LLM, incorporate them directly in your custom form_prompt() function definition.

        Returns:
            tuple[ThresholdedTrustworthyRAGScore, bool]: A tuple containing:
                - ThresholdedTrustworthyRAGScore: Quality scores for different evaluation metrics like trustworthiness
                  and response helpfulness. Each metric has a score between 0-1. It also has a boolean flag, `is_bad` indicating whether the score is below a given threshold.
                - bool: True if the response is determined to be bad based on the evaluation scores
                  and configured thresholds, False otherwise.
        """
        scores = await self._tlm_rag.score_async(
            response=response,
            query=query,
            context=context,
            prompt=prompt,
            form_prompt=form_prompt,
        )

        thresholded_scores = _update_scores_based_on_thresholds(
            scores=scores,
            thresholds=self._bad_response_thresholds,
        )

        is_bad_response = any(score_dict["is_bad"] for score_dict in thresholded_scores.values())
        return thresholded_scores, is_bad_response

    def _remediate(self, *, query: str, metadata: dict[str, Any] | None = None) -> str | None:
        """Request a SME-provided answer for this query, if one is available in Codex.

        Args:
            query (str): The user's original query to get SME-provided answer for.

        Returns:
            str | None: The SME-provided answer from Codex, or None if no answer could be found in the Codex Project.
        """
        codex_answer, _ = self._project.query(question=query, metadata=metadata)
        return codex_answer


class BadResponseThresholds(BaseModel):
    """Config for determining if a response is bad.
    Each key is an evaluation metric and the value is a threshold such that a response is considered bad whenever the corresponding evaluation score falls below the threshold.

    Default Thresholds:
        - trustworthiness: 0.7
        - response_helpfulness: 0.7
        - Any custom eval: 0.0 (if not explicitly specified in bad_response_thresholds). A threshold of 0.0 means that the associated eval is not used to determine if a response is bad, unless explicitly specified in bad_response_thresholds, but still allow for reporting of those scores.
    """

    trustworthiness: float = Field(
        description="Threshold for trustworthiness.",
        default=0.7,
        ge=0.0,
        le=1.0,
    )
    response_helpfulness: float = Field(
        description="Threshold for response helpfulness.",
        default=0.23,
        ge=0.0,
        le=1.0,
    )

    @property
    def default_threshold(self) -> float:
        """The default threshold to use when an evaluation metric's threshold is not specified. This threshold is set to 0.0."""
        return 0.0

    def get_threshold(self, eval_name: str) -> float:
        """Get threshold for an eval, if it exists.

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
