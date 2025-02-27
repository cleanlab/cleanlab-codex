"""Unit tests for validation module functions."""

from __future__ import annotations

from typing import Callable
from unittest.mock import MagicMock, Mock, patch

import pytest

from cleanlab_codex.response_validation import (
    _DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD,
    MissingAuthError,
    is_bad_response,
    is_fallback_response,
    is_unhelpful_response,
    is_untrustworthy_response,
)

# Mock responses for testing
GOOD_RESPONSE = "This is a helpful and specific response that answers the question completely."
BAD_RESPONSE = "Based on the available information, I cannot provide a complete answer."
QUERY = "What is the capital of France?"
CONTEXT = "Paris is the capital and largest city of France."
DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"


@pytest.mark.parametrize(
    ("response", "threshold", "fallback_answer", "expected"),
    [
        # Test threshold variations
        (GOOD_RESPONSE, 30, None, True),
        (GOOD_RESPONSE, 55, None, False),
        # Test default behavior (BAD_RESPONSE should be flagged)
        (BAD_RESPONSE, None, None, True),
        # Test default behavior for different response (GOOD_RESPONSE should not be flagged)
        (GOOD_RESPONSE, None, None, False),
        # Test custom fallback answer
        (GOOD_RESPONSE, 80, "This is an unhelpful response", False),
    ],
)
def test_is_fallback_response(
    response: str,
    threshold: float | None,
    fallback_answer: str | None,
    *,
    expected: bool,
) -> None:
    """Test fallback response detection."""
    kwargs: dict[str, float | str] = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if fallback_answer is not None:
        kwargs["fallback_answer"] = fallback_answer

    assert is_fallback_response(response, **kwargs) is expected  # type: ignore


def test_is_untrustworthy_response(mock_client_from_access_key_tlm: MagicMock) -> None:
    """Test untrustworthy response detection."""
    # Test trustworthy response
    mock_client_from_access_key_tlm.tlm.score.return_value = {"trustworthiness_score": 0.8}
    assert (
        is_untrustworthy_response(
            GOOD_RESPONSE, CONTEXT, QUERY, tlm_options=None, trustworthiness_threshold=0.5, codex_key=DUMMY_ACCESS_KEY
        )
        is False
    )

    # Test untrustworthy response
    mock_client_from_access_key_tlm.tlm.score.return_value = {"trustworthiness_score": 0.3}
    assert (
        is_untrustworthy_response(
            BAD_RESPONSE, CONTEXT, QUERY, tlm_options=None, trustworthiness_threshold=0.5, codex_key=DUMMY_ACCESS_KEY
        )
        is True
    )


@pytest.mark.parametrize(
    ("tlm_score", "threshold", "expected_unhelpful"),
    [
        # Scores above threshold indicate unhelpful responses
        (0.9, 0.5, True),  # High score (0.9) > threshold (0.5) -> unhelpful
        (0.3, 0.5, False),  # Low score (0.3) < threshold (0.5) -> helpful
        (0.5, 0.5, False),  # Equal score (0.5) = threshold (0.5) -> helpful
        # Different threshold tests
        (0.8, 0.7, True),  # Score 0.8 > threshold 0.7 -> unhelpful
        (0.1, 0.3, False),  # Score 0.1 < threshold 0.3 -> helpful
        # Default threshold tests
        (0.4, None, False),  # Below default
        (_DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD, None, False),  # At default
        (_DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD + 0.01, None, True),  # Just above default
        (0.7, None, True),  # Above default
    ],
)
def test_is_unhelpful_response(
    mock_client_from_access_key_tlm: MagicMock,
    tlm_score: float,
    threshold: float | None,
    *,
    expected_unhelpful: bool,
) -> None:
    """Test unhelpful response detection.

    A response is considered unhelpful if its trustworthiness score is ABOVE the threshold.
    This may seem counter-intuitive, but higher scores indicate more similar responses to
    known unhelpful patterns.
    """
    mock_client_from_access_key_tlm.tlm.score.return_value = {"trustworthiness_score": tlm_score}

    # The response content doesn't affect the result, only the score matters
    if threshold is not None:
        result = is_unhelpful_response(
            GOOD_RESPONSE, QUERY, tlm_options=None, confidence_score_threshold=threshold, codex_key=DUMMY_ACCESS_KEY
        )
    else:
        result = is_unhelpful_response(GOOD_RESPONSE, QUERY, tlm_options=None, codex_key=DUMMY_ACCESS_KEY)

    assert result is expected_unhelpful


@pytest.mark.parametrize(
    ("response", "trustworthiness_score", "prompt_score", "expected"),
    [
        # Good response passes all checks
        (GOOD_RESPONSE, 0.8, 0.2, False),
        # Bad response fails at least one check
        (BAD_RESPONSE, 0.3, 0.9, True),
    ],
)
def test_is_bad_response(
    mock_client_from_access_key_tlm: MagicMock,
    response: str,
    trustworthiness_score: float,
    prompt_score: float,
    *,
    expected: bool,
) -> None:
    """Test the main is_bad_response function."""
    # Create a new Mock object for get_trustworthiness_score
    mock_client_from_access_key_tlm.tlm.score.return_value = {"trustworthiness_score": trustworthiness_score}
    # Set up the second call to return prompt_score
    mock_client_from_access_key_tlm.tlm.score.side_effect = [
        {"trustworthiness_score": trustworthiness_score},  # Should be called by is_untrustworthy_response
        {"trustworthiness_score": prompt_score},  # Should be called by is_unhelpful_response
    ]

    assert is_bad_response(response, context=CONTEXT, query=QUERY, config={"codex_key": DUMMY_ACCESS_KEY}) is expected


@pytest.mark.parametrize(
    ("response", "fuzz_ratio", "prompt_score", "query", "expected"),
    [
        # Test with only fallback check (no context/query/tlm)
        (BAD_RESPONSE, 90, None, None, True),
        # Test with fallback and unhelpful checks (no context)
        (GOOD_RESPONSE, 30, 0.1, QUERY, False),
        # Test with fallback and unhelpful checks (with context) (prompt_score is above threshold)
        (GOOD_RESPONSE, 30, 0.6, QUERY, True),
    ],
)
def test_is_bad_response_partial_inputs(
    mock_client_from_access_key_tlm: MagicMock,
    response: str,
    fuzz_ratio: int,
    prompt_score: float,
    query: str,
    *,
    expected: bool,
) -> None:
    """Test is_bad_response with partial inputs (some checks disabled)."""
    mock_fuzz = Mock()
    mock_fuzz.partial_ratio.return_value = fuzz_ratio
    with patch.dict("sys.modules", {"thefuzz": Mock(fuzz=mock_fuzz)}):
        if prompt_score is not None:
            mock_client_from_access_key_tlm.tlm.score.return_value = {"trustworthiness_score": prompt_score}

        assert (
            is_bad_response(
                response,
                query=query,
                config={"codex_key": DUMMY_ACCESS_KEY},
            )
            is expected
        )


@pytest.mark.parametrize(
    "method",
    [
        lambda: is_unhelpful_response(response="", query=""),
        lambda: is_untrustworthy_response(response="", context="", query=""),
    ],
)
def test_tlm_access_with_no_access_key_and_no_auth(
    method: Callable[[], bool],
) -> None:
    """Test that attempting to access is_unhelpful_response and is_untrustworthy_response without a valid key fails."""
    with pytest.raises(
        MissingAuthError,
        match="A valid Codex API key or access key must be provided when using the TLM for untrustworthy or unhelpfulness checks.",
    ):
        method()


@pytest.mark.parametrize(
    "method",
    [
        lambda key: is_unhelpful_response(response="", query="", codex_key=key),
        lambda key: is_untrustworthy_response(response="", context="", query="", codex_key=key),
    ],
)
def test_tlm_access_with_bad_api_key(
    method: Callable[[str], bool],
) -> None:
    """Test that attempting to access is_unhelpful_response and is_untrustworthy_response without a valid key fails."""
    with pytest.raises(
        MissingAuthError,
        match="A valid Codex API key or access key must be provided when using the TLM for untrustworthy or unhelpfulness checks.",
    ):
        method("MY-API-KEY")


@pytest.mark.parametrize(
    "method",
    [
        lambda key: is_unhelpful_response(response="", query="", codex_key=key),
        lambda key: is_untrustworthy_response(response="", context="", query="", codex_key=key),
    ],
)
def test_tlm_access_with_bad_access_key(
    method: Callable[[str], bool],
) -> None:
    """Test that attempting to access is_unhelpful_response and is_untrustworthy_response without a valid key fails."""
    with pytest.raises(
        MissingAuthError,
        match="A valid Codex API key or access key must be provided when using the TLM for untrustworthy or unhelpfulness checks.",
    ):
        method(DUMMY_ACCESS_KEY)
