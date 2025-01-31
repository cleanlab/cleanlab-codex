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
    context: str,
    tlm: TLM, # TODO: Make this optional
    query: Optional[str] = None,
    # is_fallback_response args
    fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
    partial_ratio_threshold: int = DEFAULT_PARTIAL_RATIO_THRESHOLD,
    # is_untrustworthy_response args
    trustworthiness_threshold: float = DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    # is_unhelpful_response args
    unhelpful_trustworthiness_threshold: Optional[float] = None,
) -> bool:
    """Run a series of checks to determine if a response is bad. If any of the checks pass, return True.

    Checks:
    - Is the response too similar to a known fallback answer?
    - Is the response untrustworthy?
    - Is the response unhelpful?
    
    Args:
        response: The response to check. See `is_fallback_response`, `is_untrustworthy_response`, and `is_unhelpful_response`.
        context: The context/documents to use for answering. See `is_untrustworthy_response`.
        tlm: The TLM model to use for evaluation. See `is_untrustworthy_response` and `is_unhelpful_response`.
        query: The user's question (optional). See `is_untrustworthy_response` and `is_unhelpful_response`.
        fallback_answer: The fallback answer to compare against. See `is_fallback_response`.
        partial_ratio_threshold: The threshold for detecting fallback responses. See `is_fallback_response`.
        trustworthiness_threshold: The threshold for detecting untrustworthy responses. See `is_untrustworthy_response`.
        unhelpful_trustworthiness_threshold: The threshold for detecting unhelpful responses. See `is_unhelpful_response`.
    """
    validation_checks = [
        lambda: is_fallback_response(response, fallback_answer, threshold=partial_ratio_threshold),
        lambda: (
            is_untrustworthy_response(response, context, query, tlm, threshold=trustworthiness_threshold)
            if query is not None
            else False
        ),
        lambda: is_unhelpful_response(response, tlm, query, trustworthiness_score_threshold=unhelpful_trustworthiness_threshold)
    ]

    return any(check() for check in validation_checks)


def is_fallback_response(response: str, fallback_answer: str = DEFAULT_FALLBACK_ANSWER, threshold: int=DEFAULT_PARTIAL_RATIO_THRESHOLD) -> bool:
    """Check if a response is too similar to a known fallback answer.

    Uses fuzzy string matching to compare the response against a known fallback answer.
    Returns True if the response is similar enough to be considered unhelpful.

    Args:
        response: The response to check
        fallback_answer: A known unhelpful/fallback response to compare against
        threshold: Similarity threshold (0-100). Higher values require more similarity.
                  Default 70 means responses that are 70% or more similar are considered bad.

    Returns:
        bool: True if the response is too similar to the fallback answer, False otherwise
    """
    try:
        from thefuzz import fuzz  # type: ignore
    except ImportError:
        raise ImportError("The 'thefuzz' library is required. Please install it with `pip install thefuzz`.")

    partial_ratio = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
    return partial_ratio >= threshold

def is_untrustworthy_response(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    threshold: float = DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    format_prompt: Callable[[str, str], str] = default_format_prompt
) -> bool:
    """Check if a response is untrustworthy based on TLM's evaluation.

    Uses TLM to evaluate whether a response is trustworthy given the context and query.
    Returns True if TLM's trustworthiness score falls below the threshold, indicating
    the response may be incorrect or unreliable.

    Args:
        response: The response to check from the assistant
        context: The context information available for answering the query
        query: The user's question or request
        tlm: The TLM model to use for evaluation
        threshold: Score threshold (0.0-1.0). Lower values allow less trustworthy responses.
                  Default 0.6, meaning responses with scores less than 0.6 are considered untrustworthy.
        format_prompt: Function that takes (query, context) and returns a formatted prompt string.
                      Users should provide their RAG app's own prompt formatting function here
                      to match how their LLM is prompted.

    Returns:
        bool: True if the response is deemed untrustworthy by TLM, False otherwise
    """
    try:
        from cleanlab_studio.studio.trustworthy_language_model import TLM  # noqa: F401
    except ImportError:
        raise ImportError("The 'cleanlab_studio' library is required. Please install it with `pip install cleanlab-studio`.")

    prompt = format_prompt(query, context)
    resp = tlm.get_trustworthiness_score(prompt, response)
    score: float = resp['trustworthiness_score']
    return score < threshold


def is_unhelpful_response(response: str, tlm: TLM, query: Optional[str] = None, trustworthiness_score_threshold: Optional[float] = None) -> bool:
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
        from cleanlab_studio.studio.trustworthy_language_model import TLM  # noqa: F401
    except ImportError:
        raise ImportError("The 'cleanlab_studio' library is required. Please install it with `pip install cleanlab-studio`.")

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
        "Consider the following" +
        (f" User Query and AI Assistant Response.\n\nUser Query: {query}\n\n" if query else " AI Assistant Response.\n\n") +
        f"AI Assistant Response: {response}\n\n{question}"
    )

    output = tlm.prompt(prompt, constrain_outputs=["Yes", "No"])
    response_marked_unhelpful = output["response"].lower() == expected_unhelpful_response
    is_trustworthy = trustworthiness_score_threshold is None or (output["trustworthiness_score"] > trustworthiness_score_threshold)
    return response_marked_unhelpful and is_trustworthy
