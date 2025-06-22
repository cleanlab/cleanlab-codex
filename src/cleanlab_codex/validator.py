"""
Detect and remediate bad responses in RAG applications, by integrating Codex as-a-Backup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, Callable, Literal, Optional

from cleanlab_tlm import TrustworthyRAG

from cleanlab_codex.internal.validator import validate_thresholds
from cleanlab_codex.project import Project

if _TYPE_CHECKING:
    from codex.types.project_validate_response import ProjectValidateResponse


class Validator:
    def __init__(
        self,
        codex_access_key: str,
        eval_thresholds: Optional[dict[str, float]] = None,
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

            eval_thresholds (dict[str, float], optional): Custom thresholds (between 0 and 1) for specific evals.
                Keys should either correspond to an Eval from [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag)
                or a custom eval for your project. If not provided, project settings will be used.


        Raises:
            TypeError: If any threshold value is not a number.
            ValueError: If any threshold value is not between 0 and 1.
        """
        self._project: Project = Project.from_access_key(access_key=codex_access_key)
        if eval_thresholds is not None:
            validate_thresholds(eval_thresholds)
        self._eval_thresholds = eval_thresholds

    def validate(
        self,
        *,
        query: str,
        context: str,
        response: str,
        prompt: Optional[str] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        eval_scores: Optional[dict[str, float]] = None,
        quality_preset: Literal["best", "high", "medium", "low", "base"] = "medium",
    ) -> ProjectValidateResponse:
        """Evaluate whether the AI-generated response is bad, and if so, request an alternate expert answer.
        If no expert answer is available, this query is still logged for SMEs to answer.

        Args:
            query (str): The user query that was used to generate the response.
            context (str): The context that was retrieved from the RAG Knowledge Base and used to generate the response.
            response (str): A reponse from your LLM/RAG system.
            prompt (str, optional): Optional prompt representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
            form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other. This function should take query and context as parameters and return a formatted prompt string. If not provided, a default prompt formatter will be used. To include a system prompt or any other special instructions for your LLM, incorporate them directly in your custom form_prompt() function definition.
            metadata (dict, optional): Additional custom metadata to associate with the query logged in the Codex Project.
            eval_scores (dict[str, float], optional): Scores assessing different aspects of the RAG system. If provided, TLM Trustworthy RAG will not be used to generate scores.
            options (ProjectValidateOptions, optional): Typed dict of advanced TLM configuration options. See [TLMOptions](/tlm/api/python/tlm/#class-tlmoptions)
            quality_preset (Literal["best", "high", "medium", "low", "base"], optional): The quality preset to use for the TLM or Trustworthy RAG API.

        Returns:
            ProjectValidateResponse: A response object containing:
                - eval_scores (Dict[str, EvalScores]): Evaluation scores for the original response along with a boolean flag, `failed`,
                  indicating whether the score is below the threshold.
                - expert_answer (Optional[str]): Alternate SME-provided answer from Codex if the response was flagged as bad and
                  an answer was found in the Codex Project, or None otherwise.
                - is_bad_response (bool): True if the response is flagged as potentially bad and triggered escalation to SMEs.
        """
        formatted_prompt = prompt
        if not formatted_prompt:
            if form_prompt:
                formatted_prompt = form_prompt(query, context)
            else:
                formatted_prompt = TrustworthyRAG._default_prompt_formatter(  # noqa: SLF001
                    query, context
                )

        if not formatted_prompt:
            raise ValueError("Exactly one of prompt or form_prompt is required")  # noqa: TRY003

        return self._project.validate(
            context=context,
            prompt=formatted_prompt,
            query=query,
            response=response,
            custom_metadata=metadata,
            eval_scores=eval_scores,
            eval_thresholds=self._eval_thresholds,
            quality_preset=quality_preset,
        )
