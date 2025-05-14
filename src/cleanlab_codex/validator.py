"""
Detect and remediate bad responses in RAG applications, by integrating Codex as-a-Backup.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from cleanlab_tlm import TrustworthyRAG
from pydantic import BaseModel, Field, field_validator

from cleanlab_codex.project import Project


class Validator:
    def __init__(
        self,
        codex_access_key: str,
        bad_response_thresholds: Optional[dict[str, float]] = None,
    ):
        """Real-time detection and remediation of bad responses in RAG applications, powered by Cleanlab's TrustworthyRAG and Codex.

        This object combines Cleanlab's TrustworthyRAG evaluation scores with configurable thresholds to detect potentially bad responses
        in your RAG application. When a bad response is detected, Cleanlab automatically attempts to remediate by retrieving an expert-provided
        answer from the Codex Project you've integrated with your RAG app. If no expert answer is available,
        the corresponding query is logged in the Codex Project for SMEs to answer.

        For production, use the `validate()` method which provides a complete validation workflow including both detection and remediation.

        Args:
            codex_access_key (str): The [access key](/codex/web_tutorials/create_project/#access-keys) for a Codex project. Used to retrieve expert-provided answers
                when bad responses are detected, or otherwise log the corresponding queries for SMEs to answer.

            bad_response_thresholds (dict[str, float], optional): Detection score thresholds used to flag whether
                a response is bad or not. Each key corresponds to an Eval from [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag),
                and the value indicates a threshold (between 0 and 1) below which Eval scores are treated as detected issues. A response
                is flagged as bad if any issues are detected. If not provided or only partially provided, default thresholds will be used
                for any missing metrics. Note that if a threshold is provided for a metric, that metric must correspond to an eval
                that is configured to run (with the exception of 'trustworthiness' which is always implicitly configured). You can
                configure arbitrary evals to run, and their thresholds will use default values unless explicitly set. See
                [`BadResponseThresholds`](/codex/api/python/validator/#class-badresponsethresholds) for more details on the default values.

        Raises:
            TypeError: If any threshold value is not a number.
            ValueError: If any threshold value is not between 0 and 1.
        """
        self._project: Project = Project.from_access_key(access_key=codex_access_key)
        self._bad_response_thresholds = (
            BadResponseThresholds.model_validate(bad_response_thresholds) if bad_response_thresholds else None
        )

    def validate(
        self,
        *,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        if not prompt and not form_prompt:
            form_prompt = TrustworthyRAG._default_prompt_formatter  # noqa: SLF001

        prompt = prompt or form_prompt(query, context)

        result = self._project.validate(
            context=context,
            prompt=prompt,
            query=query,
            response=response,
            bad_response_thresholds=self._bad_response_thresholds,
            custom_metadata=metadata,
        )

        formatted_eval_scores = {
            eval_name: {
                "score": eval_scores.score,
                "is_bad": eval_scores.is_bad,
            }
            for eval_name, eval_scores in result.eval_scores.items()
        }

        return {
            "expert_answer": result.expert_answer,
            "is_bad_response": result.is_bad_response,
            **formatted_eval_scores,
        }


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
