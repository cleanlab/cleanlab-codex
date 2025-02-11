"""
This module provides validation functions for evaluating LLM responses and determining if they should be replaced with Codex-generated alternatives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypedDict, Union, cast

from cleanlab_codex.utils.errors import MissingDependencyError
from cleanlab_codex.utils.prompt import default_format_prompt

if TYPE_CHECKING:
    try:
        from cleanlab_studio.studio.trustworthy_language_model import TLM  # type: ignore
    except ImportError:
        from typing import Any, Dict, Protocol, Sequence

        class _TLMProtocol(Protocol):
            def get_trustworthiness_score(
                self,
                prompt: Union[str, Sequence[str]],
                response: Union[str, Sequence[str]],
                **kwargs: Any,
            ) -> Dict[str, Any]: ...

            def prompt(
                self,
                prompt: Union[str, Sequence[str]],
                /,
                **kwargs: Any,
            ) -> Dict[str, Any]: ...

        TLM = _TLMProtocol


DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."
DEFAULT_PARTIAL_RATIO_THRESHOLD = 70
DEFAULT_TRUSTWORTHINESS_THRESHOLD = 0.5


class BadResponseDetectionConfig(TypedDict, total=False):
    """Configuration for bad response detection functions.
    See get_bad_response_config() for default values.

    Attributes:
        fallback_answer: Known unhelpful response to compare against
        partial_ratio_threshold: Similarity threshold (0-100). Higher values require more similarity
        trustworthiness_threshold: Score threshold (0.0-1.0). Lower values allow less trustworthy responses
        format_prompt: Function to format (query, context) into a prompt string
        unhelpfulness_confidence_threshold: Optional confidence threshold (0.0-1.0) for unhelpful classification
        tlm: TLM model to use for evaluation (required for untrustworthiness and unhelpfulness checks)
    """

    # Fallback check config
    fallback_answer: str
    partial_ratio_threshold: int

    # Untrustworthy check config
    trustworthiness_threshold: float
    format_prompt: Callable[[str, str], str]

    # Unhelpful check config
    unhelpfulness_confidence_threshold: Optional[float]

    # Shared config (for untrustworthiness and unhelpfulness checks)
    tlm: Optional[TLM]


def get_bad_response_config() -> BadResponseDetectionConfig:
    """Get the default configuration for bad response detection functions.

    Returns:
        BadResponseDetectionConfig: Default configuration for bad response detection functions
    """
    return {
        "fallback_answer": DEFAULT_FALLBACK_ANSWER,
        "partial_ratio_threshold": DEFAULT_PARTIAL_RATIO_THRESHOLD,
        "trustworthiness_threshold": DEFAULT_TRUSTWORTHINESS_THRESHOLD,
        "format_prompt": default_format_prompt,
        "unhelpfulness_confidence_threshold": None,
        "tlm": None,
    }


def is_bad_response(
    response: str,
    *,
    context: Optional[str] = None,
    query: Optional[str] = None,
    config: Optional[BadResponseDetectionConfig] = None,
) -> bool:
    """Run a series of checks to determine if a response is bad.

    If any check detects an issue (i.e. fails), the function returns True, indicating the response is bad.

    This function runs three possible validation checks:
    1. **Fallback check**: Detects if response is too similar to a known fallback answer.
    2. **Untrustworthy check**: Assesses response trustworthiness based on the given context and query.
    3. **Unhelpful check**: Predicts if the response adequately answers the query or not, in a useful way.

    Note:
    Each validation check runs conditionally based on whether the required arguments are provided.
    As soon as any validation check fails, the function returns True.

    Args:
        response: The response to check.
        context: Optional context/documents used for answering. Required for untrustworthy check.
        query: Optional user question. Required for untrustworthy and unhelpful checks.
        config: Optional, typed dictionary of configuration parameters. See <_BadReponseConfig> for details.

    Returns:
        bool: True if any validation check fails, False if all pass.
    """
    default_cfg = get_bad_response_config()
    cfg: BadResponseDetectionConfig
    cfg = {**default_cfg, **(config or {})}

    validation_checks: list[Callable[[], bool]] = []

    # All required inputs are available for checking fallback responses
    validation_checks.append(
        lambda: is_fallback_response(
            response,
            cfg["fallback_answer"],
            threshold=cfg["partial_ratio_threshold"],
        )
    )

    can_run_untrustworthy_check = query is not None and context is not None and cfg["tlm"] is not None
    if can_run_untrustworthy_check:
        # The if condition guarantees these are not None
        validation_checks.append(
            lambda: is_untrustworthy_response(
                response=response,
                context=cast(str, context),
                query=cast(str, query),
                tlm=cfg["tlm"],
                trustworthiness_threshold=cfg["trustworthiness_threshold"],
                format_prompt=cfg["format_prompt"],
            )
        )

    can_run_unhelpful_check = query is not None and cfg["tlm"] is not None
    if can_run_unhelpful_check:
        validation_checks.append(
            lambda: is_unhelpful_response(
                response=response,
                query=cast(str, query),
                tlm=cfg["tlm"],
                trustworthiness_score_threshold=cast(float, cfg["unhelpfulness_confidence_threshold"]),
            )
        )

    return any(check() for check in validation_checks)


def is_fallback_response(
    response: str, fallback_answer: str = DEFAULT_FALLBACK_ANSWER, threshold: int = DEFAULT_PARTIAL_RATIO_THRESHOLD
) -> bool:
    """Check if a response is too similar to a known fallback answer.

    Uses fuzzy string matching to compare the response against a known fallback answer.
    Returns True if the response is similar enough to be considered unhelpful.

    Args:
        response: The response to check.
        fallback_answer: A known unhelpful/fallback response to compare against.
        threshold: Similarity threshold (0-100). Higher values require more similarity.
                  Default 70 means responses that are 70% or more similar are considered bad.

    Returns:
        bool: True if the response is too similar to the fallback answer, False otherwise
    """
    try:
        from thefuzz import fuzz  # type: ignore
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "thefuzz",
            package_url="https://github.com/seatgeek/thefuzz",
        ) from e

    partial_ratio: int = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
    return bool(partial_ratio >= threshold)


def is_untrustworthy_response(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    trustworthiness_threshold: float = DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    format_prompt: Callable[[str, str], str] = default_format_prompt,
) -> bool:
    """Check if a response is untrustworthy.

    Uses TLM to evaluate whether a response is trustworthy given the context and query.
    Returns True if TLM's trustworthiness score falls below the threshold, indicating
    the response may be incorrect or unreliable.

    Args:
        response: The response to check from the assistant
        context: The context information available for answering the query
        query: The user's question or request
        tlm: The TLM model to use for evaluation
        trustworthiness_threshold: Score threshold (0.0-1.0). Lower values allow less trustworthy responses.
                  Default 0.5, meaning responses with scores less than 0.5 are considered untrustworthy.
        format_prompt: Function that takes (query, context) and returns a formatted prompt string.
                      Users should provide their RAG app's own prompt formatting function here
                      to match how their LLM is prompted.

    Returns:
        bool: True if the response is deemed untrustworthy by TLM, False otherwise
    """
    try:
        from cleanlab_studio import Studio  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_studio",
            package_name="cleanlab-studio",
            package_url="https://github.com/cleanlab/cleanlab-studio",
        ) from e

    prompt = format_prompt(query, context)
    result = tlm.get_trustworthiness_score(prompt, response)
    score: float = result["trustworthiness_score"]
    return score < trustworthiness_threshold


def is_unhelpful_response(
    response: str,
    query: str,
    tlm: TLM,
    trustworthiness_score_threshold: Optional[float] = None,
) -> bool:
    """Check if a response is unhelpful by asking TLM to evaluate it.

    Uses TLM to evaluate whether a response is helpful by asking it to make a Yes/No judgment.
    The evaluation considers both the TLM's binary classification of helpfulness and its
    confidence score. Returns True only if TLM classifies the response as unhelpful AND
    is sufficiently confident in that assessment (if a threshold is provided).

    Args:
        response: The response to check
        query: User query that will be used to evaluate if the response is helpful
        tlm: The TLM model to use for evaluation
        trustworthiness_score_threshold: Optional confidence threshold (0.0-1.0)
                                       If provided and the response is marked as unhelpful,
                                       the confidence score must exceed this threshold for
                                       the response to be considered truly unhelpful.

    Returns:
        bool: True if TLM determines the response is unhelpful with sufficient confidence,
              False otherwise
    """
    try:
        from cleanlab_studio import Studio  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_studio",
            package_name="cleanlab-studio",
            package_url="https://github.com/cleanlab/cleanlab-studio",
        ) from e

    # The question and expected "unhelpful" response are linked:
    # - When asking "is helpful?" -> "no" means unhelpful
    # - When asking "is unhelpful?" -> "yes" means unhelpful
    question = (
        "Does the AI Assistant Response seem unhelpful? "
        "Things that are not helpful include answers that:\n"
        "- Are not useful, incomplete, incorrect, uncertain or unclear.\n"
        "- Abstain or refuse to answer the question\n"
        "- Statements which are similar to 'I don't know', 'Sorry', or 'No information available'.\n"
        "- Leave the original question unresolved\n"
        "- Are irrelevant to the question\n"
        "Answer Yes/No only."
    )
    expected_unhelpful_response = "yes"

    prompt = (
        "Consider the following User Query and AI Assistant Response.\n\n"
        f"User Query: {query}\n\n"
        f"AI Assistant Response: {response}\n\n"
        f"{question}"
    )

    output = tlm.prompt(prompt, constrain_outputs=["Yes", "No"])
    response_marked_unhelpful = output["response"].lower() == expected_unhelpful_response
    is_trustworthy = trustworthiness_score_threshold is None or (
        output["trustworthiness_score"] > trustworthiness_score_threshold
    )
    return response_marked_unhelpful and is_trustworthy
