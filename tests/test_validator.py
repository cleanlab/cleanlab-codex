from typing import Generator
from unittest.mock import Mock, patch

import pytest
from codex.types.project_validate_response import EvalScores, ProjectValidateResponse

from cleanlab_codex.validator import Validator


@pytest.fixture
def mock_project() -> Generator[Mock, None, None]:
    with patch("cleanlab_codex.validator.Project") as mock:
        mock_obj = Mock()
        mock_obj.validate.return_value = ProjectValidateResponse(
            is_bad_response=True,
            expert_answer=None,
            eval_scores={
                "response_helpfulness": EvalScores(score=0.95, failed=False),
                "trustworthiness": EvalScores(score=0.5, failed=True),
            },
        )
        mock.from_access_key.return_value = mock_obj
        yield mock


@pytest.fixture
def mock_project_with_custom_thresholds() -> Generator[Mock, None, None]:
    with patch("cleanlab_codex.validator.Project") as mock:
        mock_obj = Mock()
        mock_obj.validate.return_value = ProjectValidateResponse(
            is_bad_response=False,
            expert_answer=None,
            eval_scores={
                "response_helpfulness": EvalScores(score=0.95, failed=False),
                "trustworthiness": EvalScores(score=0.5, failed=False),
            },
        )
        mock.from_access_key.return_value = mock_obj
        yield mock


class TestValidator:
    def test_init(self, mock_project: Mock) -> None:
        Validator(codex_access_key="test")

        # Verify Project was initialized with access key
        mock_project.from_access_key.assert_called_once_with(access_key="test")

    def test_validate(self, mock_project: Mock) -> None:  # noqa: ARG002
        validator = Validator(codex_access_key="test")

        result = validator.validate(query="test query", context="test context", response="test response")

        # Verify expected result structure
        assert result.is_bad_response is True
        assert result.expert_answer is None

    def test_validate_expert_answer(self, mock_project: Mock) -> None:
        validator = Validator(codex_access_key="test", eval_thresholds={"trustworthiness": 1.0})
        mock_project.from_access_key.return_value.query.return_value = (None, None)
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result.expert_answer is None

        # Setup mock project query response
        mock_project.from_access_key.return_value.validate.return_value = ProjectValidateResponse(
            is_bad_response=True,
            expert_answer="expert answer",
            eval_scores={
                "response_helpfulness": EvalScores(score=0.95, failed=False),
                "trustworthiness": EvalScores(score=0.5, failed=True),
            },
        )
        # Basically any response will be flagged as untrustworthy
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result.expert_answer == "expert answer"

    def test_user_provided_thresholds(self, mock_project_with_custom_thresholds: Mock) -> None:
        """
        Test with user-provided thresholds.
        Higher trustworthiness threshold should not produce a bad response and extra threshold should not raise ValueError
        """
        validator = Validator(
            codex_access_key="test",
            eval_thresholds={"trustworthiness": 0.4, "non_existent_metric": 0.5},
        )
        mock_project_with_custom_thresholds.from_access_key.assert_called_once_with(access_key="test")
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result.is_bad_response is False
        assert result.expert_answer is None

    def test_default_thresholds(self, mock_project: Mock) -> None:
        # Test with default thresholds (eval_thresholds is None)
        validator = Validator(codex_access_key="test")
        mock_project.from_access_key.assert_called_once_with(access_key="test")
        assert validator._eval_thresholds is None  # noqa: SLF001
