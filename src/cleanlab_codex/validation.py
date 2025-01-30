"""
This module provides validation functions for checking if an LLM response is unhelpful.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cleanlab_studio.studio.trustworthy_language_model import TLM


def is_bad_response(response: str, fallback_answer: str, threshold: int = 70) -> bool:
    """Check if a response is too similar to a known fallback/unhelpful answer.

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
        from thefuzz import fuzz
    except ImportError:
        raise ImportError("The 'thefuzz' library is required. Please install it with `pip install thefuzz`.")

    partial_ratio = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
    return partial_ratio >= threshold

def is_bad_response_contains_phrase(response: str, fallback_responses: list[str]) -> bool:
    """Check if a response is unhelpful by looking for known fallback phrases.

    Uses simple substring matching to check if the response contains any known fallback phrases
    that indicate the response is unhelpful (e.g. "I cannot help with that", "I don't know").
    Returns True if any fallback phrase is found in the response.

    Args:
        response: The response to check from the assistant
        fallback_responses: List of known fallback phrases that indicate an unhelpful response.
                          The check is case-insensitive.

    Returns:
        bool: True if the response contains any fallback phrase, False otherwise
    """
    return any(phrase.lower() in response.lower() for phrase in fallback_responses)

def is_bad_response_untrustworthy(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    threshold: float = 0.6,
    # TODO: Optimize prompt template
    prompt_template: str = "Using the following Context, provide a helpful answer to the Query.\n\n Context:\n{context}\n\n Query: {query}",
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
        prompt_template: Template for formatting the evaluation prompt. Must contain {context}
                       and {query} placeholders.

    Returns:
        bool: True if the response is deemed untrustworthy by TLM, False otherwise
    """
    prompt = prompt_template.format(context=context, query=query)
    resp = tlm.get_trustworthiness_score(prompt, response)
    score: float = resp['trustworthiness_score']
    return score < threshold

# TLM Binary Classification
def is_bad_response_unhelpful(response: str, tlm: TLM, query: Optional[str] = None, trustworthiness_score_threshold: Optional[float] = None) -> bool:
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
    if query is None:
        prompt = (
            "Consider the following AI Assistant Response.\n\n"
            f"AI Assistant Response: {response}\n\n"
            "Is the AI Assistant Response helpful? Remember that abstaining from responding is not helpful. Answer Yes/No only."
        )
    else:
        prompt = (
            "Consider the following User Query and AI Assistant Response.\n\n"
            f"User Query: {query}\n\n"
            f"AI Assistant Response: {response}\n\n"
            "Is the AI Assistant Response helpful? Remember that abstaining from responding is not helpful. Answer Yes/No only."
        )
    output = tlm.prompt(prompt, constrain_outputs=["Yes", "No"])
    response_marked_unhelpful = output["response"].lower() == "no"
    # TODO: Decide if we should keep the trustworthiness score threshold.
    is_trustworthy = trustworthiness_score_threshold is None or (output["trustworthiness_score"] > trustworthiness_score_threshold)
    return response_marked_unhelpful and is_trustworthy
