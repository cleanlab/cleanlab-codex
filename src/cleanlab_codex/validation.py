"""
This module provides validation functions for checking if an LLM response is unhelpful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from cleanlab_codex.utils.prompt import default_format_prompt

if TYPE_CHECKING:
    from cleanlab_studio.studio.trustworthy_language_model import TLM  # type: ignore


DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."
DEFAULT_PARTIAL_RATIO_THRESHOLD = 70
DEFAULT_TRUSTWORTHINESS_THRESHOLD = 0.5


def is_bad_response(
    response: str,
    *,
    context: Optional[str] = None,
    query: Optional[str] = None,
    tlm: Optional[TLM] = None,
    # is_fallback_response args
    fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
    partial_ratio_threshold: int = DEFAULT_PARTIAL_RATIO_THRESHOLD,
    # is_untrustworthy_response args
    trustworthiness_threshold: float = DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    format_prompt: Callable[[str, str], str] = default_format_prompt,
    # is_unhelpful_response args
    unhelpful_trustworthiness_threshold: Optional[float] = None,
) -> bool:
    """Run a series of checks to determine if a response is bad. If any of the checks pass, return True.

    This function runs three possible validation checks:
    1. Fallback check: Detects if response is too similar to known fallback answers.
    2. Untrustworthy check: Evaluates response trustworthiness given context and query.
    3. Unhelpful check: Evaluates if response is helpful for the given query.

    Args:
        response: The response to check.
        context: Optional context/documents used for answering. Required for untrustworthy check.
        query: Optional user question. Required for untrustworthy and unhelpful checks.
        tlm: Optional TLM model for evaluation. Required for untrustworthy and unhelpful checks.
        
        # Fallback check parameters
        fallback_answer: Known unhelpful response to compare against.
        partial_ratio_threshold: Similarity threshold (0-100). Higher values require more similarity.
        
        # Untrustworthy check parameters
        trustworthiness_threshold: Score threshold (0.0-1.0). Lower values allow less trustworthy responses.
        format_prompt: Function to format (query, context) into a prompt string.
        
        # Unhelpful check parameters
        unhelpful_trustworthiness_threshold: Optional confidence threshold (0.0-1.0) for unhelpful classification.

    Returns:
        bool: True if any validation check fails, False if all pass.
    """

    validation_checks = []

    # All required inputs are available for checking fallback responses
    validation_checks.append(
        lambda: is_fallback_response(response, fallback_answer, threshold=partial_ratio_threshold)
    )

    can_run_untrustworthy_check = all(x is not None for x in (query, context, tlm))
    if can_run_untrustworthy_check:
        assert tlm is not None
        assert query is not None
        assert context is not None
        validation_checks.append(
            lambda: is_untrustworthy_response(
                response=response,
                context=context,
                query=query,
                tlm=tlm,
                threshold=trustworthiness_threshold,
                format_prompt=format_prompt,
            )
        )

    can_run_unhelpful_check = query is not None and tlm is not None
    if can_run_unhelpful_check:
        assert tlm is not None
        assert query is not None
        validation_checks.append(
            lambda: is_unhelpful_response(
                response=response,
                tlm=tlm,
                query=query,
                trustworthiness_score_threshold=unhelpful_trustworthiness_threshold,
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
        error_msg = "The 'thefuzz' library is required. Please install it with `pip install thefuzz`."
        raise ImportError(error_msg) from e

    partial_ratio: int = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
    return bool(partial_ratio >= threshold)


def is_untrustworthy_response(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    threshold: float = DEFAULT_TRUSTWORTHINESS_THRESHOLD,
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
        threshold: Score threshold (0.0-1.0). Lower values allow less trustworthy responses.
                  Default 0.5, meaning responses with scores less than 0.5 are considered untrustworthy.
        format_prompt: Function that takes (query, context) and returns a formatted prompt string.
                      Users should provide their RAG app's own prompt formatting function here
                      to match how their LLM is prompted.

    Returns:
        bool: True if the response is deemed untrustworthy by TLM, False otherwise
    """
    try:
        from cleanlab_studio import Studio  # type: ignore # noqa: F401
    except ImportError as e:
        error_msg = "The 'cleanlab_studio' library is required. Please install it with `pip install cleanlab-studio`."
        raise ImportError(error_msg) from e

    prompt = format_prompt(query, context)
    result = tlm.get_trustworthiness_score(prompt, response)
    score: float = result["trustworthiness_score"]
    return score < threshold


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
        response: The response to check from the assistant
        tlm: The TLM model to use for evaluation
        query: Optional user query to provide context for evaluating helpfulness.
              If provided, TLM will assess if the response helpfully answers this query.
        trustworthiness_score_threshold: Optional confidence threshold (0.0-1.0).
                                       If provided, responses are only marked unhelpful if TLM's
                                       confidence score exceeds this threshold.

    Returns:
        bool: True if TLM determines the response is unhelpful with sufficient confidence,
              False otherwise
    """
    try:
        from cleanlab_studio import Studio  # type: ignore # noqa: F401
    except ImportError as e:
        error_msg = "The 'cleanlab_studio' library is required. Please install it with `pip install cleanlab-studio`."
        raise ImportError(error_msg) from e

    # The question and expected "unhelpful" response are linked:
    # - When asking "is helpful?" -> "no" means unhelpful
    # - When asking "is unhelpful?" -> "yes" means unhelpful
    question = (
        "Is the AI Assistant Response unhelpful? "
        "Unhelpful responses include answers that:\n"
        "- Are not useful, incomplete, incorrect, uncertain or unclear.\n"
        "- Abstain or refuse to answer the question\n"
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
