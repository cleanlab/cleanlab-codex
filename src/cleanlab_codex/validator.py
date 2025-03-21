"""
Leverage Cleanlab's Evals and Codex to detect and remediate bad responses in RAG applications.
"""

from __future__ import annotations

from typing import Any, Optional, cast

from cleanlab_codex.internal.validator import (
    IsBadResponseConfig,
    get_default_evaluations,
    get_default_trustworthyrag_config,
)
from cleanlab_codex.internal.validator import is_bad_response as _is_bad_response
from cleanlab_codex.project import Project
from cleanlab_codex.utils.errors import MissingDependencyError

try:
    from cleanlab_tlm import TrustworthyRAG
    from cleanlab_tlm.utils.rag import TrustworthyRAGScore
except ImportError as e:
    raise MissingDependencyError(
        import_name=e.name or "cleanlab-tlm",
        package_url="https://github.com/cleanlab/cleanlab-tlm",
    ) from e


class Validator:
    def __init__(
        self,
        codex_access_key: str,
        tlm_api_key: Optional[str] = None,
        trustworthy_rag_config: Optional[dict[str, Any]] = None,
        is_bad_response_config: Optional[dict[str, float]] = None,
    ):
        """Evaluates the quality of responses generated in RAG applications and remediates them if needed.

        This object combines Cleanlab's various Evals with thresholding to detect bad responses and remediates them with Codex.

        Args:
            codex_access_key (str): The [access key](/codex/web_tutorials/create_project/#access-keys) for a Codex project.
            tlm_api_key (Optional[str]): The API key for [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag).
            trustworthy_rag_config (Optional[dict[str, Any]]): The constructor arguments for [TrustworthyRAG](/tlm/api/python/utils.rag/#class-trustworthyrag).
            is_bad_response_config (Optional[dict[str, float]]): The configuration for determining if a response is bad.
        """
        trustworthy_rag_config = trustworthy_rag_config or get_default_trustworthyrag_config()
        if tlm_api_key is not None:
            trustworthy_rag_config["api_key"] = tlm_api_key
        self._is_bad_response_config = IsBadResponseConfig.model_validate(is_bad_response_config or {})

        self._project: Project = Project.from_access_key(access_key=codex_access_key)

        trustworthy_rag_config.setdefault("evals", get_default_evaluations())
        self._tlm_rag = TrustworthyRAG(**trustworthy_rag_config)

    def validate(self, query: str, context: str, response: str) -> dict[str, Any]:
        """Validate the response quality and generate an alternative response if needed.

        Args:
            query (str): The user's original query.
            context (str): The context provided to generate the response.
            response (str): The response to evaluate.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'is_bad_response': True if the response is determined to be bad, False otherwise.
                - 'expert_answer': The alternative response from Codex, or None if no response could be fetched from Codex.
                - Other evaluation metrics from TrustworthyRAG.
        """
        scores, is_bad_response = self.detect(query, context, response)
        expert_answer = None
        if is_bad_response:
            expert_answer = self.remediate(query)

        return {
            "is_bad_response": is_bad_response,
            "expert_answer": expert_answer,
            **scores,
        }

    def detect(self, query: str, context: str, response: str) -> tuple[TrustworthyRAGScore, bool]:
        """Evaluate the response quality using TrustworthyRAG and determine if it is a bad response.

        Args:
            query (str): The user's original query.
            context (str): The context provided to generate the response.
            response (str): The response to evaluate.

        Returns:
            tuple[TrustworthyRAGScore, bool]: A tuple containing:
                - TrustworthyRAGScore: Quality scores for different evaluation metrics like trustworthiness
                  and response helpfulness. Each metric has a score between 0-1.
                - bool: True if the response is determined to be bad based on the evaluation scores
                  and configured thresholds, False otherwise.
        """
        scores = cast(TrustworthyRAGScore, self._tlm_rag.score(response=response, query=query, context=context))
        _config = (
            IsBadResponseConfig.model_validate(self._is_bad_response_config) if self._is_bad_response_config else None
        )
        is_bad_response = _is_bad_response(scores, _config)
        return scores, is_bad_response

    def remediate(self, query: str) -> str | None:
        """Queries Codex to get an alternative response when the original response is determined to be bad.

        Args:
            query (str): The user's original query to get an alternative response for.

        Returns:
            str | None: The alternative response from Codex, or None if no response could be fetched from Codex.
        """
        codex_answer, _ = self._project.query(question=query)
        return codex_answer
