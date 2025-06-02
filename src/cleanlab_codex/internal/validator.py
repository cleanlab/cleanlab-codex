from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

from cleanlab_tlm.tlm import TLMResponse
from cleanlab_tlm.utils.rag import Eval, TrustworthyRAGScore, get_default_evals

from cleanlab_codex.types.validator import ThresholdedTrustworthyRAGScore

if TYPE_CHECKING:
    from cleanlab_tlm import TLM

    from cleanlab_codex.validator import BadResponseThresholds


"""Evaluation metrics (excluding trustworthiness) that are used to determine if a response is bad."""
DEFAULT_EVAL_METRICS = ["response_helpfulness"]

# Simple mappings for is_bad keys
_SCORE_TO_IS_BAD_KEY = {
    "trustworthiness": "is_not_trustworthy",
    "query_ease": "is_not_query_easy",
    "response_helpfulness": "is_not_response_helpful",
    "context_sufficiency": "is_not_context_sufficient",
}

REWRITE_QUERY = """Given a conversational Message History and a Query, rewrite the Query to be a self-contained question. If the query is already self contained, return it as-is.

Query: {query}

--
Message History: \n{messages}

--
Remember, return the Query as-is except in cases where the Query is missing key words or has content that should be additionally clarified."""
REWRITE_QUERY_TRUSTWORTHINESS_THRESHOLD = 0.5


def get_default_evaluations() -> list[Eval]:
    """Get the default evaluations for the TrustworthyRAG.

    Note:
        This excludes trustworthiness, which is automatically computed by TrustworthyRAG.
    """
    return [evaluation for evaluation in get_default_evals() if evaluation.name in DEFAULT_EVAL_METRICS]


def get_default_trustworthyrag_config() -> dict[str, Any]:
    """Get the default configuration for the TrustworthyRAG."""
    return {
        "options": {
            "log": ["explanation"],
        },
    }


def get_default_tlm_config() -> dict[str, Any]:
    """Get the default configuration for the TLM."""

    return {
        "quality_preset": "medium",
        "verbose": False,
        "options": {"model": "gpt-4.1-nano"},
    }


def update_scores_based_on_thresholds(
    scores: TrustworthyRAGScore | Sequence[TrustworthyRAGScore], thresholds: BadResponseThresholds
) -> ThresholdedTrustworthyRAGScore:
    """Adds a `is_bad` flag to the scores dictionaries based on the thresholds."""

    # Helper function to check if a score is bad
    def is_bad(score: Optional[float], threshold: float) -> bool:
        return score is not None and score < threshold

    if isinstance(scores, Sequence):
        raise NotImplementedError("Batching is not supported yet.")

    thresholded_scores = {}
    for eval_name, score_dict in scores.items():
        thresholded_scores[eval_name] = {
            **score_dict,
            "is_bad": is_bad(score_dict["score"], thresholds.get_threshold(eval_name)),
        }
    return cast(ThresholdedTrustworthyRAGScore, thresholded_scores)


def process_score_metadata(scores: ThresholdedTrustworthyRAGScore, thresholds: BadResponseThresholds) -> dict[str, Any]:
    """Process scores into metadata format with standardized keys.

    Args:
        scores: The ThresholdedTrustworthyRAGScore containing evaluation results
        thresholds: The BadResponseThresholds configuration

    Returns:
        dict: A dictionary containing evaluation scores and their corresponding metadata
    """
    metadata: dict[str, Any] = {}

    # Process scores and add to metadata
    for metric, score_data in scores.items():
        metadata[metric] = score_data["score"]

        # Add is_bad flags with standardized naming
        is_bad_key = _SCORE_TO_IS_BAD_KEY.get(metric, f"is_not_{metric}")
        metadata[is_bad_key] = score_data["is_bad"]

        # Special case for trustworthiness explanation
        if metric == "trustworthiness" and "log" in score_data and "explanation" in score_data["log"]:
            metadata["explanation"] = score_data["log"]["explanation"]

    # Add thresholds to metadata
    thresholds_dict = thresholds.model_dump()
    for metric in {k for k in scores if k not in thresholds_dict}:
        thresholds_dict[metric] = thresholds.get_threshold(metric)
    metadata["thresholds"] = thresholds_dict

    # TODO: Remove this as the backend can infer this from the is_bad flags
    metadata["label"] = _get_label(metadata)

    return metadata


def _get_label(metadata: dict[str, Any]) -> str:
    def is_bad(metric: str) -> bool:
        return bool(metadata.get(_SCORE_TO_IS_BAD_KEY[metric], False))

    if is_bad("context_sufficiency"):
        return "search_failure"
    if is_bad("response_helpfulness") or is_bad("query_ease"):
        return "unhelpful"
    if is_bad("trustworthiness"):
        return "hallucination"
    return "other_issues"


def validate_messages(messages: Optional[list[dict[str, Any]]] = None) -> None:
    """Validate the format of messages based on OpenAI."""

    if messages is None:
        return
    if not isinstance(messages, list):
        raise TypeError("Messages must be a list of dictionaries.")  # noqa: TRY003
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise TypeError("Each message must be a dictionary containing 'role' and 'content' keys.")  # noqa: TRY003
        if not isinstance(message["content"], str):
            raise TypeError("Message content must be a string.")  # noqa: TRY003


def prompt_tlm_for_rewrite_query(query: str, messages: list[dict[str, Any]], tlm: TLM) -> TLMResponse:
    """Given the query and message history, prompt the TLM for a response that could possibly be self contained.
    If the tlm call fails, then the original query is returned.
    """

    messages_str = ""
    for message in messages:
        messages_str += f"{message['role']}: {message['content']}\n"

    response = tlm.prompt(
        REWRITE_QUERY.format(
            query=query,
            messages=messages_str,
        )
    )

    if response is None:
        return TLMResponse(response=query, trustworthiness_score=1.0)
    return cast(TLMResponse, response)
