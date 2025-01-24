"""
This module provides validation functions for checking if an LLM response is inadequate/unhelpful.
The default implementation checks for common fallback phrases, but alternative implementations
are provided below as examples that can be adapted for specific needs.
"""


def is_bad_response(response: str) -> bool:
    """
    Default implementation that checks for common fallback phrases from LLM assistants.

    NOTE: YOU SHOULD MODIFY THIS METHOD YOURSELF.
    """
    return basic_validator(response)


def basic_validator(response: str) -> bool:
    """Basic implementation that checks for common fallback phrases from LLM assistants.

    Args:
        response: The response from the assistant

    Returns:
        bool: True if the response appears to be a fallback/inadequate response
    """
    partial_fallback_responses = [
        "Based on the available information",
        "I cannot provide a complete answer to this question",
        # Add more substrings here to improve the recall of the check
    ]
    return any(
        partial_fallback_response.lower() in response.lower()
        for partial_fallback_response in partial_fallback_responses
    )


# Alternative Implementations
# ---------------------------
# The following implementations are provided as examples and inspiration.
# They should be adapted to your specific needs.


# Fuzzy String Matching
"""
from thefuzz import fuzz

def fuzzy_match_validator(response: str, fallback_answer: str, threshold: int = 70) -> bool:
    partial_ratio = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
    return partial_ratio >= threshold
"""

# TLM Score Thresholding
"""
from cleanlab_studio import Studio

studio = Studio("<API_KEY>")
tlm = studio.TLM()

def tlm_score_validator(response: str, context: str, query: str, tlm: TLM, threshold: float = 0.5) -> bool:
    prompt = f"Context: {context}\n\n Query: {query}\n\n Query: {query}"
    resp = tlm.get_trustworthiness_score(prompt, response)
    score = resp['trustworthiness_score']
    return score < threshold
"""

# TLM Binary Classification
"""
from typing import Optional

from cleanlab_studio import Studio

studio = Studio("<API_KEY>")
tlm = studio.TLM()

def tlm_binary_validator(response: str, tlm: TLM, query: Optional[str] = None) -> bool:
    if query is None:
        prompt = f"Here is a response from an AI assistant: {response}\n\n Is it helpful? Answer Yes/No only."
    else:
        prompt = f"Here is a response from an AI assistant: {response}\n\n Considering the following query: {query}\n\n Is the response helpful? Answer Yes/No only."
    output = tlm.prompt(prompt)
    return output["response"].lower() == "no"
"""
