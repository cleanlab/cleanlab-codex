from typing import Generator
from unittest.mock import Mock, patch

import pytest
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
        assert thresholds.get_threshold("trustworthiness") == 0.5
        assert thresholds.get_threshold("response_helpfulness") == 0.5

    def test_unspecified_threshold(self) -> None:
        thresholds = BadResponseThresholds()
        assert thresholds.get_threshold("unspecified_threshold") == 0.5

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
        mock.from_access_key.return_value = Mock()
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


class TestValidator:
    def test_init(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:
        Validator(codex_access_key="test")

        # Verify Project was initialized with access key
        mock_project.from_access_key.assert_called_once_with(access_key="test")

        # Verify TrustworthyRAG was initialized with default config
        mock_trustworthy_rag.assert_called_once()

    def test_init_with_tlm_api_key(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        Validator(codex_access_key="test", tlm_api_key="tlm-key")

        # Verify TrustworthyRAG was initialized with API key
        config = mock_trustworthy_rag.call_args[1]
        assert config["api_key"] == "tlm-key"

    def test_init_with_config_conflict(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        with pytest.raises(ValueError, match="Cannot specify both tlm_api_key and api_key in trustworthy_rag_config"):
            Validator(codex_access_key="test", tlm_api_key="tlm-key", trustworthy_rag_config={"api_key": "config-key"})

    def test_validate(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        validator = Validator(codex_access_key="test")

        result = validator.validate(query="test query", context="test context", response="test response")

        # Verify TrustworthyRAG.score was called
        mock_trustworthy_rag.return_value.score.assert_called_once_with(
            response="test response", query="test query", context="test context", prompt=None, form_prompt=None
        )

        # Verify expected result structure
        assert result["is_bad_response"] is False
        assert result["expert_answer"] is None

        eval_metrics = ["trustworthiness", "response_helpfulness"]
        for metric in eval_metrics:
            assert metric in result
            assert "score" in result[metric]
            assert "is_bad" in result[metric]

    def test_validate_expert_answer(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        # Setup mock project query response
        mock_project.from_access_key.return_value.query.return_value = ("expert answer", None)

        # Basically any response will be flagged as untrustworthy
        validator = Validator(codex_access_key="test", bad_response_thresholds={"trustworthiness": 1.0})
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result["expert_answer"] == "expert answer"

        mock_project.from_access_key.return_value.query.return_value = (None, None)
        result = validator.validate(query="test query", context="test context", response="test response")
        assert result["expert_answer"] is None

    def test_detect(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        validator = Validator(codex_access_key="test")

        scores, is_bad = validator.detect(query="test query", context="test context", response="test response")

        # Verify scores match mock return value
        assert scores["trustworthiness"]["score"] == 0.8
        assert scores["response_helpfulness"]["score"] == 0.7
        assert not is_bad  # Since mock scores are above default thresholds

    def test_remediate(self, mock_project: Mock, mock_trustworthy_rag: Mock) -> None:  # noqa: ARG002
        # Setup mock project query response
        mock_project.from_access_key.return_value.query.return_value = ("expert answer", None)

        validator = Validator(codex_access_key="test")
        result = validator._remediate("test query")

        # Verify project.query was called
        mock_project.from_access_key.return_value.query.assert_called_once_with(question="test query")
        assert result == "expert answer"
