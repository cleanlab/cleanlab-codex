"""
Validation functions for evaluating LLM responses and determining if they should be replaced with Codex-generated alternatives.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field

from cleanlab_codex.internal.utils import generate_pydantic_model_docstring
from cleanlab_codex.types.tlm import TLM
from cleanlab_codex.utils.errors import MissingDependencyError
from cleanlab_codex.utils.prompt import default_format_prompt

_DEFAULT_FALLBACK_ANSWER: str = (
    "Based on the available information, I cannot provide a complete answer to this question."
)
_DEFAULT_FALLBACK_SIMILARITY_THRESHOLD: int = 70
_DEFAULT_TRUSTWORTHINESS_THRESHOLD: float = 0.5
_DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD: float = 0.5

Query = str
Context = str
Prompt = str


class BadResponseDetectionConfig(BaseModel):
    """Configuration for bad response detection functions.

    Used by [`is_bad_response`](#function-is_bad_response) function to which passes values to corresponding downstream validation checks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fallback check config
    fallback_answer: str = Field(
        default=_DEFAULT_FALLBACK_ANSWER,
        description="Known unhelpful response to compare against",
    )
    fallback_similarity_threshold: int = Field(
        default=_DEFAULT_FALLBACK_SIMILARITY_THRESHOLD,
        description="Fuzzy matching similarity threshold (0-100). Higher values mean responses must be more similar to fallback_answer to be considered bad.",
        ge=0,
        le=100,
    )

    # Untrustworthy check config
    trustworthiness_threshold: float = Field(
        default=_DEFAULT_TRUSTWORTHINESS_THRESHOLD,
        description="Score threshold (0.0-1.0). Lower values allow less trustworthy responses.",
        ge=0.0,
        le=1.0,
    )
    format_prompt: Callable[[Query, Context], Prompt] = Field(
        default=default_format_prompt,
        description="Function to format (query, context) into a prompt string.",
    )

    # Unhelpful check config
    unhelpfulness_confidence_threshold: float = Field(
        default=_DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD,
        description="Confidence threshold (0.0-1.0) for unhelpful classification.",
        ge=0.0,
        le=1.0,
    )

    # Shared config (for untrustworthiness and unhelpfulness checks)
    tlm: Optional[TLM] = Field(
        default=None,
        description="TLM model to use for evaluation (required for untrustworthiness and unhelpfulness checks).",
    )


# hack to generate better documentation for help.cleanlab.ai
BadResponseDetectionConfig.__doc__ = f"""
{BadResponseDetectionConfig.__doc__}

{generate_pydantic_model_docstring(BadResponseDetectionConfig, BadResponseDetectionConfig.__name__)}
"""

_DEFAULT_CONFIG = BadResponseDetectionConfig()


def is_bad_response(
    response: str,
    *,
    context: Optional[str] = None,
    query: Optional[str] = None,
    config: Union[BadResponseDetectionConfig, Dict[str, Any]] = _DEFAULT_CONFIG,
) -> bool:
    """Run a series of checks to determine if a response is bad.

    If any check detects an issue (i.e. fails), the function returns `True`, indicating the response is bad.

    This function runs three possible validation checks:
    1. **Fallback check**: Detects if response is too similar to a known fallback answer.
    2. **Untrustworthy check**: Assesses response trustworthiness based on the given context and query.
    3. **Unhelpful check**: Predicts if the response adequately answers the query or not, in a useful way.

    Note: Each validation check runs conditionally based on whether the required arguments are provided.
    As soon as any validation check fails, the function returns `True`.

    Args:
        response (str): The response to check.
        context (str, optional): Optional context/documents used for answering. Required for untrustworthy check.
        query (str, optional): Optional user question. Required for untrustworthy and unhelpful checks.
        config (BadResponseDetectionConfig, optional): Optional, configuration parameters for validation checks. See [BadResponseDetectionConfig](#class-badresponsedetectionconfig) for details. If not provided, default values will be used.

    Returns:
        bool: `True` if any validation check fails, `False` if all pass.
    """
    config = BadResponseDetectionConfig.model_validate(config)

    validation_checks: list[Callable[[], bool]] = []

    # All required inputs are available for checking fallback responses
    validation_checks.append(
        lambda: is_fallback_response(
            response,
            config.fallback_answer,
            threshold=config.fallback_similarity_threshold,
        )
    )

    can_run_untrustworthy_check = query is not None and context is not None and config.tlm is not None
    if can_run_untrustworthy_check:
        # The if condition guarantees these are not None
        validation_checks.append(
            lambda: is_untrustworthy_response(
                response=response,
                context=cast(str, context),
                query=cast(str, query),
                tlm=cast(TLM, config.tlm),
                trustworthiness_threshold=config.trustworthiness_threshold,
                format_prompt=config.format_prompt,
            )
        )

    can_run_unhelpful_check = query is not None and config.tlm is not None
    if can_run_unhelpful_check:
        validation_checks.append(
            lambda: is_unhelpful_response(
                response=response,
                query=cast(str, query),
                tlm=cast(TLM, config.tlm),
                confidence_score_threshold=config.unhelpfulness_confidence_threshold,
            )
        )

    return any(check() for check in validation_checks)


def is_fallback_response(
    response: str,
    fallback_answer: str = _DEFAULT_FALLBACK_ANSWER,
    threshold: int = _DEFAULT_FALLBACK_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if a response is too similar to a known fallback answer.

    Uses fuzzy string matching to compare the response against a known fallback answer.
    Returns `True` if the response is similar enough to the fallback answer to be considered unhelpful.

    Args:
        response (str): The response to check.
        fallback_answer (str): A known unhelpful/fallback response to compare against.
        threshold (int): Similarity threshold (0-100) above which a response is considered to match the fallback answer.
                Higher values require more similarity. Default 70 means responses that are 70% or more similar are considered bad.

    Returns:
        bool: `True` if the response is too similar to the fallback answer, `False` otherwise.
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
    trustworthiness_threshold: float = _DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    format_prompt: Callable[[str, str], str] = default_format_prompt,
) -> bool:
    """Check if a response is untrustworthy.

    Uses [TLM](/tlm) to evaluate whether a response is trustworthy given the context and query.
    Returns `True` if TLM's trustworthiness score falls below the threshold, indicating
    the response may be incorrect or unreliable.

    Args:
        response (str): The response to check from the assistant.
        context (str): The context information available for answering the query.
        query (str): The user's question or request.
        tlm (TLM): The TLM model to use for evaluation.
        trustworthiness_threshold (float): Score threshold (0.0-1.0) under which a response is considered untrustworthy.
                  Lower values allow less trustworthy responses. Default 0.5 means responses with scores less than 0.5 are considered untrustworthy.
        format_prompt (Callable[[str, str], str]): Function that takes (query, context) and returns a formatted prompt string.
                      Users should provide the prompt formatting function for their RAG application here so that the response can
                      be evaluated using the same prompt that was used to generate the response.

    Returns:
        bool: `True` if the response is deemed untrustworthy by TLM, `False` otherwise.
    """
    try:
        from cleanlab_tlm import TLM
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_tlm",
            package_name="cleanlab-tlm",
            package_url="https://github.com/cleanlab/cleanlab-tlm",
        ) from e

    prompt = format_prompt(query, context)
    result = tlm.get_trustworthiness_score(prompt, response)
    score: float = result["trustworthiness_score"]
    return score < trustworthiness_threshold


def is_unhelpful_response(
    response: str,
    query: str,
    tlm: TLM,
    confidence_score_threshold: float = _DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD,
) -> bool:
    """Check if a response is unhelpful by asking [TLM](/tlm) to evaluate it.

    Uses TLM to evaluate whether a response is helpful by asking it to make a Yes/No judgment.
    The evaluation considers both the TLM's binary classification of helpfulness and its
    confidence score. Returns `True` only if TLM classifies the response as unhelpful AND
    is sufficiently confident in that assessment (if a threshold is provided).

    Args:
        response (str): The response to check.
        query (str): User query that will be used to evaluate if the response is helpful.
        tlm (TLM): The TLM model to use for evaluation.
        confidence_score_threshold (float): Confidence threshold (0.0-1.0) above which a response is considered unhelpful.
                                       E.g. if confidence_score_threshold is 0.5, then responses with scores higher than 0.5 are considered unhelpful.

    Returns:
        bool: `True` if TLM determines the response is unhelpful with sufficient confidence,
              `False` otherwise.
    """
    try:
        from cleanlab_tlm import TLM
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_tlm",
            package_name="cleanlab-tlm",
            package_url="https://github.com/cleanlab/cleanlab-tlm",
        ) from e

    # IMPORTANT: The current implementation couples three things that must stay in sync:
    # 1. The question phrasing ("is unhelpful?")
    # 2. The expected_unhelpful_response ("Yes")
    # 3. The threshold logic (score > threshold)
    #
    # If changing the question to "is helpful?", you would need to:
    # If changing the question to "is helpful?", you would need to either:
    # Option A:
    #   1. Change expected_unhelpful_response to "No"
    #   2. Keep the threshold logic as: score > threshold
    # Option B:
    #   1. Keep expected_unhelpful_response as "Yes"
    #   2. Invert the threshold logic to: score < threshold
    # In either case:
    #   Consider adjusting the default threshold value since confidence scores
    #      might have different distributions for positive vs negative questions
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
    expected_unhelpful_response = "Yes"

    prompt = (
        "Consider the following User Query and AI Assistant Response.\n\n"
        f"User Query: {query}\n\n"
        f"AI Assistant Response: {response}\n\n"
        f"{question}"
    )

    output = tlm.get_trustworthiness_score(prompt, response=expected_unhelpful_response, constrain_outputs=["Yes", "No"])

    # Current implementation assumes question is phrased to expect "Yes" for unhelpful responses
    # Changing the question would require restructuring this logic and potentially adjusting
    # the threshold value in BadResponseDetectionConfig
    return float(output["trustworthiness_score"]) > confidence_score_threshold
