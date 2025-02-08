"""Unit tests for validation module functions."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from cleanlab_codex.validation import (
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


@pytest.fixture
def mock_tlm() -> Mock:
    """Create a mock TLM instance."""
    mock = Mock()
    # Configure default return values
    mock.get_trustworthiness_score.return_value = {"trustworthiness_score": 0.8}
    mock.prompt.return_value = {"response": "No", "trustworthiness_score": 0.9}
    return mock


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


def test_is_untrustworthy_response(mock_tlm: Mock) -> None:
    """Test untrustworthy response detection."""
    # Test trustworthy response
    mock_tlm.get_trustworthiness_score.return_value = {"trustworthiness_score": 0.8}
    assert is_untrustworthy_response(GOOD_RESPONSE, CONTEXT, QUERY, mock_tlm, threshold=0.5) is False

    # Test untrustworthy response
    mock_tlm.get_trustworthiness_score.return_value = {"trustworthiness_score": 0.3}
    assert is_untrustworthy_response(BAD_RESPONSE, CONTEXT, QUERY, mock_tlm, threshold=0.5) is True


@pytest.mark.parametrize(
    ("response", "tlm_response", "tlm_score", "threshold", "expected"),
    [
        # Test helpful response
        (GOOD_RESPONSE, "No", 0.9, 0.5, False),
        # Test unhelpful response
        (BAD_RESPONSE, "Yes", 0.9, 0.5, True),
        # Test unhelpful response but low trustworthiness score
        (BAD_RESPONSE, "Yes", 0.3, 0.5, False),
        # Test without threshold - Yes prediction
        (BAD_RESPONSE, "Yes", 0.3, None, True),
        (GOOD_RESPONSE, "Yes", 0.3, None, True),
        # Test without threshold - No prediction
        (BAD_RESPONSE, "No", 0.3, None, False),
        (GOOD_RESPONSE, "No", 0.3, None, False),
    ],
)
def test_is_unhelpful_response(
    mock_tlm: Mock,
    response: str,
    tlm_response: str,
    tlm_score: float,
    threshold: float | None,
    *,
    expected: bool,
) -> None:
    """Test unhelpful response detection."""
    mock_tlm.prompt.return_value = {"response": tlm_response, "trustworthiness_score": tlm_score}
    assert is_unhelpful_response(response, QUERY, mock_tlm, trustworthiness_score_threshold=threshold) is expected


@pytest.mark.parametrize(
    ("response", "trustworthiness_score", "prompt_response", "prompt_score", "expected"),
    [
        # Good response passes all checks
        (GOOD_RESPONSE, 0.8, "No", 0.9, False),
        # Bad response fails at least one check
        (BAD_RESPONSE, 0.3, "Yes", 0.9, True),
    ],
)
def test_is_bad_response(
    mock_tlm: Mock,
    response: str,
    trustworthiness_score: float,
    prompt_response: str,
    prompt_score: float,
    *,
    expected: bool,
) -> None:
    """Test the main is_bad_response function."""
    mock_tlm.get_trustworthiness_score.return_value = {"trustworthiness_score": trustworthiness_score}
    mock_tlm.prompt.return_value = {"response": prompt_response, "trustworthiness_score": prompt_score}

    assert (
        is_bad_response(
            response,
            context=CONTEXT,
            query=QUERY,
            tlm=mock_tlm,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("response", "fuzz_ratio", "prompt_response", "prompt_score", "query", "tlm", "expected"),
    [
        # Test with only fallback check (no context/query/tlm)
        (BAD_RESPONSE, 90, None, None, None, None, True),
        # Test with fallback and unhelpful checks (no context)
        (GOOD_RESPONSE, 30, "No", 0.9, QUERY, "mock_tlm", False),
    ],
)
def test_is_bad_response_partial_inputs(
    mock_tlm: Mock,
    response: str,
    fuzz_ratio: int,
    prompt_response: str,
    prompt_score: float,
    query: str,
    tlm: Mock,
    *,
    expected: bool,
) -> None:
    """Test is_bad_response with partial inputs (some checks disabled)."""
    mock_fuzz = Mock()

    with patch.dict("sys.modules", {"thefuzz": Mock(fuzz=mock_fuzz)}):
        if prompt_response is not None:
            mock_tlm.prompt.return_value = {"response": prompt_response, "trustworthiness_score": prompt_score}
            tlm = mock_tlm

        assert (
            is_bad_response(
                response,
                query=query,
                tlm=tlm,
            )
            is expected
        )
