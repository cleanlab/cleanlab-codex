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
from cleanlab_codex.types.response_validation import (
    AggregatedResponseValidationResult,
    SingleResponseValidationResult,
)
from cleanlab_codex.types.tlm import TLM
from cleanlab_codex.utils.errors import MissingDependencyError
from cleanlab_codex.utils.prompt import default_format_prompt

_DEFAULT_FALLBACK_ANSWER: str = (
    "Based on the available information, I cannot provide a complete answer to this question."
)
_DEFAULT_FALLBACK_SIMILARITY_THRESHOLD: float = 0.7
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
    fallback_similarity_threshold: float = Field(
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
) -> AggregatedResponseValidationResult:
    """Run a series of checks to determine if a response is bad.

    The function returns a `AggregatedResponseValidationResult` object containing results from multiple validation checks.
    If any check fails (detects an issue), the AggregatedResponseValidationResult will evaluate to True when used in a boolean context.
    This means code like `if is_bad_response(...)` will enter the if-block when problems are detected.

    For example:
    ```python
    is_bad = is_bad_response(...)
    if is_bad:  # True if any validation check failed
        print("Response had issues")
        # Access detailed results through is_bad.results
    ```
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
        AggregatedResponseValidationResult: A AggregatedResponseValidationResult object containing the results of the validation checks.
    """
    config = BadResponseDetectionConfig.model_validate(config)

    validation_checks: list[Callable[[], SingleResponseValidationResult]] = []

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

    results = []
    # Run all checks and collect results, until one fails
    for check in (check() for check in validation_checks):
        results.append(check)
        if check.fails_check:
            break

    return AggregatedResponseValidationResult(name="bad", results=results)


def is_fallback_response(
    response: str,
    fallback_answer: str = _DEFAULT_FALLBACK_ANSWER,
    threshold: float = _DEFAULT_FALLBACK_SIMILARITY_THRESHOLD,
) -> SingleResponseValidationResult:
    """Check if a response is too similar to a known fallback answer.

    Uses fuzzy string matching to compare the response against a known fallback answer.
    Returns `True` if the response is similar enough to the fallback answer to be considered unhelpful.

    Args:
        response (str): The response to check.
        fallback_answer (str): A known unhelpful/fallback response to compare against.
        threshold (float): Similarity threshold (0-1.0) above which a response is considered to match the fallback answer.
                Higher values require more similarity. Default 0.7 means responses that are 70% or more similar are considered bad.

    Returns:
        SingleResponseValidationResult: A SingleResponseValidationResult object containing the results of the validation checks.
    """

    score: float = score_fallback_response(response, fallback_answer)
    return SingleResponseValidationResult(
        name="fallback",
        fails_check=score >= threshold,
        score={"similarity_score": score},
        metadata={"threshold": threshold},
    )


def score_fallback_response(
    response: str,
    fallback_answer: str = _DEFAULT_FALLBACK_ANSWER,
) -> float:
    """Score a response against a known fallback answer, based on how similar they are using fuzzy string matching.

    Args:
        response (str): The response to check.
        fallback_answer (str): A known unhelpful/fallback response to compare against.

    Returns:
        float: The score of the response, between 0.0 and 1.0.
    """
    try:
        from thefuzz import fuzz  # type: ignore
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "thefuzz",
            package_url="https://github.com/seatgeek/thefuzz",
        ) from e

    return float(fuzz.partial_ratio(fallback_answer.lower(), response.lower())) / 100


def is_untrustworthy_response(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    trustworthiness_threshold: float = _DEFAULT_TRUSTWORTHINESS_THRESHOLD,
    format_prompt: Callable[[str, str], str] = default_format_prompt,
) -> SingleResponseValidationResult:
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
        SingleResponseValidationResult: A SingleResponseValidationResult object containing the results of the validation checks.
    """
    score: float = score_untrustworthy_response(
        response=response,
        context=context,
        query=query,
        tlm=tlm,
        format_prompt=format_prompt,
    )
    return SingleResponseValidationResult(
        name="untrustworthy",
        fails_check=score < trustworthiness_threshold,
        score={"trustworthiness_score": score},
        metadata={"trustworthiness_threshold": trustworthiness_threshold},
    )


def score_untrustworthy_response(
    response: str,
    context: str,
    query: str,
    tlm: TLM,
    format_prompt: Callable[[str, str], str] = default_format_prompt,
) -> float:
    """Scores a response's untrustworthiness using [TLM](/tlm), given a context and query.

    Args:
        response (str): The response to check from the assistant.
        context (str): The context information available for answering the query.
        query (str): The user's question or request.
        tlm (TLM): The TLM model to use for evaluation.
        format_prompt (Callable[[str, str], str]): Function that takes (query, context) and returns a formatted prompt string.
                    Users should provide the prompt formatting function for their RAG application here so that the response can
                    be evaluated using the same prompt that was used to generate the response.

    Returns:
        float: The score of the response, between 0.0 and 1.0.
    """
    try:
        from cleanlab_tlm import TLM  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_tlm",
            package_name="cleanlab-tlm",
            package_url="https://github.com/cleanlab/cleanlab-tlm",
        ) from e
    prompt = format_prompt(query, context)
    result = tlm.get_trustworthiness_score(prompt, response)
    return float(result["trustworthiness_score"])


def is_unhelpful_response(
    response: str,
    query: str,
    tlm: TLM,
    confidence_score_threshold: float = _DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD,
) -> SingleResponseValidationResult:
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
        SingleResponseValidationResult: A SingleResponseValidationResult object containing the results of the validation checks.
    """
    score: float = score_unhelpful_response(response, query, tlm)

    # Current implementation assumes question is phrased to expect "Yes" for unhelpful responses
    # Changing the question would require restructuring this logic and potentially adjusting
    # the threshold value in BadResponseDetectionConfig
    return SingleResponseValidationResult(
        name="unhelpful",
        fails_check=score > confidence_score_threshold,
        score={"confidence_score": score},
        metadata={"confidence_score_threshold": confidence_score_threshold},
    )


def score_unhelpful_response(
    response: str,
    query: str,
    tlm: TLM,
) -> float:
    """Scores a response's unhelpfulness using [TLM](/tlm), given a query.

    Args:
        response (str): The response to check.
        query (str): User query that will be used to evaluate if the response is helpful.
        tlm (TLM): The TLM model to use for evaluation.

    Returns:
        float: The score of the response, between 0.0 and 1.0.
    """
    try:
        from cleanlab_tlm import TLM  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            import_name=e.name or "cleanlab_tlm",
            package_name="cleanlab-tlm",
            package_url="https://github.com/cleanlab/cleanlab-tlm",
        ) from e

    # IMPORTANT: The current implementation couples three things that must stay in sync:
    # 1. The question phrasing ("is unhelpful?")
    # 2. The expected_unhelpful_response ("Yes")
    # 3. The threshold logic (score > threshold), see `is_unhelpful_response` for details
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
    result = tlm.get_trustworthiness_score(prompt, response=expected_unhelpful_response)
    return float(result["trustworthiness_score"])
