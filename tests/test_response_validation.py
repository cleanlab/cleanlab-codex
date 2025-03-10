"""Unit tests for validation module functions."""

from __future__ import annotations

from typing import Any, Generator, Sequence, Union
from unittest.mock import Mock, patch

import pytest

from cleanlab_codex.response_validation import (
    _DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD,
    is_bad_response,
    is_fallback_response,
    is_unhelpful_response,
    is_untrustworthy_response,
    score_fallback_response,
    score_unhelpful_response,
    score_untrustworthy_response,
)
from cleanlab_codex.types.response_validation import (
    AggregatedResponseValidationResult,
    SingleResponseValidationResult,
)
from cleanlab_codex.types.tlm import TLMConfig, TLMPromptResponse, TLMScoreResponse

# Mock responses for testing
GOOD_RESPONSE = "This is a helpful and specific response that answers the question completely."
BAD_RESPONSE = "Based on the available information, I cannot provide a complete answer."
QUERY = "What is the capital of France?"
CONTEXT = "Paris is the capital and largest city of France."


class MockTLM(Mock):
    _trustworthiness_score: float = 0.8
    _response: str = "No"

    @property
    def trustworthiness_score(self) -> float:
        return self._trustworthiness_score

    @trustworthiness_score.setter
    def trustworthiness_score(self, value: float) -> None:
        self._trustworthiness_score = value

    @property
    def response(self) -> str:
        return self._response

    @response.setter
    def response(self, value: str) -> None:
        self._response = value

    def get_trustworthiness_score(
        self,
        prompt: Union[str, Sequence[str]],  # noqa: ARG002
        response: Union[str, Sequence[str]],  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> TLMScoreResponse:
        return TLMScoreResponse(trustworthiness_score=self._trustworthiness_score)

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> TLMPromptResponse:
        return TLMPromptResponse(trustworthiness_score=self._trustworthiness_score, response=self._response)


@pytest.fixture
def mock_tlm_client() -> Generator[Mock, None, None]:
    with patch("cleanlab_codex.response_validation.TLM.from_config") as mock_tlm_from_config:
        mock_tlm = MockTLM()
        mock_tlm_from_config.return_value = mock_tlm
        yield mock_tlm


@pytest.mark.parametrize(
    ("response", "threshold", "fallback_answer", "expected"),
    [
        # Test threshold variations
        (GOOD_RESPONSE, 0.3, None, True),
        (GOOD_RESPONSE, 0.55, None, False),
        # Test default behavior (BAD_RESPONSE should be flagged)
        (BAD_RESPONSE, None, None, True),
        # Test default behavior for different response (GOOD_RESPONSE should not be flagged)
        (GOOD_RESPONSE, None, None, False),
        # Test custom fallback answer
        (GOOD_RESPONSE, 0.8, "This is an unhelpful response", False),
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

    assert bool(is_fallback_response(response, **kwargs)) == expected  # type: ignore


def test_is_untrustworthy_response(mock_tlm_client: Mock) -> None:
    """Test untrustworthy response detection."""
    # Test trustworthy response
    mock_tlm_client.trustworthiness_score = 0.8
    assert not bool(
        is_untrustworthy_response(
            GOOD_RESPONSE,
            CONTEXT,
            QUERY,
            mock_tlm_client,
            trustworthiness_threshold=0.5,
        )
    )

    # Test untrustworthy response
    mock_tlm_client.trustworthiness_score = 0.3
    assert bool(is_untrustworthy_response(BAD_RESPONSE, CONTEXT, QUERY, mock_tlm_client, trustworthiness_threshold=0.5))


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
        (
            _DEFAULT_UNHELPFULNESS_CONFIDENCE_THRESHOLD + 0.01,
            None,
            True,
        ),  # Just above default
        (0.7, None, True),  # Above default
    ],
)
def test_is_unhelpful_response(
    mock_tlm_client: Mock,
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
    mock_tlm_client.trustworthiness_score = tlm_score

    # The response content doesn't affect the result, only the score matters
    if threshold is not None:
        result = is_unhelpful_response(GOOD_RESPONSE, QUERY, mock_tlm_client, confidence_score_threshold=threshold)
    else:
        result = is_unhelpful_response(GOOD_RESPONSE, QUERY, mock_tlm_client)

    assert bool(result) == expected_unhelpful


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
    mock_tlm_client: Mock,
    response: str,
    trustworthiness_score: float,
    prompt_score: float,
    *,
    expected: bool,
) -> None:
    """Test the main is_bad_response function."""
    # Create a new Mock object for get_trustworthiness_score
    mock_tlm_client.get_trustworthiness_score = Mock()
    # Set up the second call to return prompt_score
    mock_tlm_client.get_trustworthiness_score.side_effect = [
        # Should be called by is_untrustworthy_response
        TLMScoreResponse(trustworthiness_score=trustworthiness_score),
        # Should be called by is_unhelpful_response
        TLMScoreResponse(trustworthiness_score=prompt_score),
    ]

    assert (
        bool(
            is_bad_response(
                response,
                context=CONTEXT,
                query=QUERY,
                config={"tlm": mock_tlm_client},
            )
        )
        == expected
    )


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
    mock_tlm_client: Mock,
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
            mock_tlm_client.get_trustworthiness_score = Mock(
                return_value=TLMScoreResponse(trustworthiness_score=prompt_score)
            )

        assert bool(is_bad_response(response, query=query)) == expected


@pytest.mark.parametrize(
    ("response", "fallback_answer", "expected"),
    [
        ("This is a test response", "This is a test response", 1.0),  # exact match
        ("abcd", "Abcd", 1.0),  # same response, different case
        ("This is a test response", "This is a test answer", 0.86),  # similar response
        (
            "This is a test response",
            "A totally different fallback answer",
            0.39,
        ),  # different response
        ("abcd", "efgh", 0.0),  # no characters in common
        ("abcd", "dcba", 0.4),  # reverse order
        ("don't know", "I don't know", 1.0),  # partial match
        (
            "I don't know",
            "don't know",
            1.0,
        ),  # partial match, response longer than fallback
    ],
)
def test_score_fallback_response(response: str, fallback_answer: str, expected: int) -> None:
    assert score_fallback_response(response, fallback_answer) == expected


@pytest.mark.parametrize(
    ("tlm_score"),
    [
        (0.5),
        (0.8),
        (0.3),
        (0.0),
    ],
)
def test_score_untrustworthy_response(mock_tlm_client: Mock, tlm_score: float) -> None:
    """Test score_untrustworthy_response function."""
    mock_tlm_client.get_trustworthiness_score = Mock(return_value=TLMScoreResponse(trustworthiness_score=tlm_score))
    assert (
        score_untrustworthy_response(
            response="A response",
            context="Some context",
            query="A query",
            tlm_config=mock_tlm_client,
        )
        == tlm_score
    )


@pytest.mark.parametrize(
    ("tlm_score"),
    [
        (0.5),
        (0.8),
        (0.3),
        (0.0),
    ],
)
def test_score_unhelpful_response(mock_tlm_client: Mock, tlm_score: float) -> None:
    """Test score_unhelpful_response function."""
    mock_tlm_client.get_trustworthiness_score = Mock(return_value=TLMScoreResponse(trustworthiness_score=tlm_score))
    assert (
        score_unhelpful_response(
            response="A response",
            query="A query",
            tlm_config=TLMConfig(),
        )
        == tlm_score
    )


class TestSingleResponseValidationResult:
    def test_single_response_validation_result_init(self) -> None:
        for name in ["fallback", "untrustworthy", "unhelpful"]:
            for fails_check in [True, False]:
                result = SingleResponseValidationResult(
                    name=name,  # type: ignore
                    fails_check=fails_check,
                    score={"similarity_score": 0.5},
                    metadata={"context": "Some context"},
                )
                assert result.name == name
                assert result.fails_check == fails_check
                assert result.score == {"similarity_score": 0.5}
                assert result.metadata == {"context": "Some context"}

    def test_bool_conversion(self) -> None:
        result = SingleResponseValidationResult(
            name="fallback",
            fails_check=True,
            score={"similarity_score": 0.5},
            metadata={"context": "Some context"},
        )
        assert bool(result)
        result.fails_check = False
        assert not bool(result)

    def test_repr(self) -> None:
        result = SingleResponseValidationResult(
            name="fallback",
            fails_check=True,
            score={"similarity_score": 0.5},
            metadata={"context": "Some context"},
        )
        assert "Failed Check" in repr(result)

        result.fails_check = False
        assert "Passed Check" in repr(result)

    def test_invalid_name(self) -> None:
        from pydantic_core import ValidationError

        for invalid_name in ["bad", "invalid"]:
            with pytest.raises(ValidationError):
                SingleResponseValidationResult(
                    name=invalid_name,  # type: ignore
                    fails_check=True,
                    score={},
                    metadata={},
                )


class TestAggregatedResponseValidationResult:
    @pytest.fixture
    def fallback_result(self) -> SingleResponseValidationResult:
        return SingleResponseValidationResult(
            name="fallback",
            fails_check=True,
            score={"similarity_score": 0.5},
            metadata={"context": "Some context"},
        )

    def test_aggregated_response_validation_result_init(self, fallback_result: SingleResponseValidationResult) -> None:
        result = AggregatedResponseValidationResult(name="bad", results=[fallback_result])
        assert result.name == "bad"
        assert result.fails_check == fallback_result.fails_check
        assert result.results == [fallback_result]

    def test_bool_conversion(self, fallback_result: SingleResponseValidationResult) -> None:
        result = AggregatedResponseValidationResult(name="bad", results=[fallback_result])
        assert bool(result)

        failed_single_response_result = SingleResponseValidationResult(
            name="fallback",
            fails_check=False,
            score={"similarity_score": 0.5},
            metadata={"context": "Some context"},
        )
        result = AggregatedResponseValidationResult(
            name="bad",
            results=[failed_single_response_result],
        )
        assert not bool(result)

    def test_invalid_name(self) -> None:
        from pydantic_core import ValidationError

        for invalid_name in ["invalid", "fallback", "untrustworthy", "unhelpful"]:
            with pytest.raises(ValidationError):
                AggregatedResponseValidationResult(
                    name=invalid_name,  # type: ignore
                    results=[],
                )
