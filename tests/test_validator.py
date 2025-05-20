from typing import Generator
from unittest.mock import Mock, patch

import pytest
from codex.types.project_validate_response import EvalScores, ProjectValidateResponse
from pydantic import ValidationError

from cleanlab_codex.validator import BadResponseThresholds, Validator


class TestBadResponseThresholds:
    def test_get_threshold(self) -> None:
        thresholds = BadResponseThresholds(
            trustworthiness=0.5,
            response_helpfulness=0.5,
        )
        assert thresholds.get_threshold("trustworthiness") == 0.5
        assert thresholds.get_threshold("response_helpfulness") == 0.5

    def test_default_threshold(self) -> None:
        thresholds = BadResponseThresholds()
        assert thresholds.get_threshold("trustworthiness") == 0.7
        assert thresholds.get_threshold("response_helpfulness") == 0.23

    def test_unspecified_threshold(self) -> None:
        thresholds = BadResponseThresholds()
        assert thresholds.get_threshold("unspecified_threshold") == 0.0

    def test_threshold_value(self) -> None:
        thresholds = BadResponseThresholds(valid_threshold=0.3)  # type: ignore
        assert thresholds.get_threshold("valid_threshold") == 0.3
        assert thresholds.valid_threshold == 0.3  # type: ignore

    def test_invalid_threshold_value(self) -> None:
        with pytest.raises(ValidationError):
            BadResponseThresholds(trustworthiness=1.1)

        with pytest.raises(ValidationError):
            BadResponseThresholds(response_helpfulness=-0.1)

    def test_invalid_threshold_type(self) -> None:
        with pytest.raises(ValidationError):
            BadResponseThresholds(trustworthiness="not a number")  # type: ignore


@pytest.fixture
def mock_project() -> Generator[Mock, None, None]:
    with patch("cleanlab_codex.validator.Project") as mock:
        mock_obj = Mock()
        mock_obj.validate.return_value = ProjectValidateResponse(
            is_bad_response=True,
            expert_answer=None,
            eval_scores={
                "response_helpfulness": EvalScores(score=0.95, is_bad=False),
                "trustworthiness": EvalScores(score=0.5, is_bad=True),
            },
        )
        mock.from_access_key.return_value = mock_obj
        yield mock


@pytest.fixture
def mock_trustworthy_rag() -> Generator[Mock, None, None]:
    mock = Mock()
    mock.score.return_value = {
        "trustworthiness": {"score": 0.8, "is_bad": False},
        "response_helpfulness": {"score": 0.7, "is_bad": False},
    }
    eval_mock = Mock()
    eval_mock.name = "response_helpfulness"
    mock.get_evals.return_value = [eval_mock]
    with patch("cleanlab_codex.validator.TrustworthyRAG") as mock_class:
        mock_class.return_value = mock
        yield mock_class


def assert_threshold_equal(validator: Validator, eval_name: str, threshold: float) -> None:
    assert validator._bad_response_thresholds.get_threshold(eval_name) == threshold  # noqa: SLF001


class TestValidator:
    def test_init(self, mock_project: Mock) -> None:
        Validator(codex_access_key="test")

        # Verify Project was initialized with access key
        mock_project.from_access_key.assert_called_once_with(access_key="test")

    def test_validate(self, mock_project: Mock) -> None:  # noqa: ARG002
        validator = Validator(codex_access_key="test")

        result = validator.validate(query="test query", context="test context", response="test response")

        # Verify expected result structure
        assert result["is_bad_response"] is True
        assert result["expert_answer"] is None

        eval_metrics = ["trustworthiness", "response_helpfulness"]
        for metric in eval_metrics:
            assert metric in result
            assert "score" in result[metric]
            assert "is_bad" in result[metric]

    def test_validate_expert_answer(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        validator = Validator(codex_access_key="test", bad_response_thresholds={"trustworthiness": 1.0})

        mock_project.from_access_key.return_value.query.return_value = (None, None)
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result["expert_answer"] is None

        # Setup mock project query response
        mock_project.from_access_key.return_value.validate.return_value = ProjectValidateResponse(
            is_bad_response=True,
            expert_answer="expert answer",
            eval_scores={
                "response_helpfulness": EvalScores(score=0.95, is_bad=False),
                "trustworthiness": EvalScores(score=0.5, is_bad=True),
            },
        )

        # Basically any response will be flagged as untrustworthy
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result["expert_answer"] == "expert answer"

    def test_user_provided_thresholds(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        # Test with user-provided thresholds that match evals
        validator = Validator(codex_access_key="test", bad_response_thresholds={"trustworthiness": 0.6})
        assert_threshold_equal(validator, "trustworthiness", 0.6)
        assert_threshold_equal(validator, "response_helpfulness", 0.23)

        # extra threshold should not raise ValueError
        Validator(codex_access_key="test", bad_response_thresholds={"non_existent_metric": 0.5})

    def test_default_thresholds(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        # Test with default thresholds (bad_response_thresholds is None)
        validator = Validator(codex_access_key="test")
        assert validator._bad_response_thresholds is None  # noqa: SLF001

    def test_edge_cases(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        # Note, the `"evals"` field should not be a list of strings in practice, but an Eval from cleanlab_tlm
        # For testing purposes, we can just

        # Test with empty bad_response_thresholds
        validator = Validator(codex_access_key="test", bad_response_thresholds={})
        assert validator._bad_response_thresholds is None  # noqa: SLF001
