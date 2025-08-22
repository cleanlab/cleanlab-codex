import uuid
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, call

import pytest
from codex import AuthenticationError
from codex.types.project_create_params import Config
from codex.types.project_validate_response import EvalScores, ProjectValidateResponse
from codex.types.projects.access_key_retrieve_project_id_response import (
    AccessKeyRetrieveProjectIDResponse,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam

from cleanlab_codex.project import MissingProjectError, Project

FAKE_PROJECT_ID = str(uuid.uuid4())
FAKE_USER_ID = "Test User"
FAKE_ORGANIZATION_ID = "Test Organization"
FAKE_PROJECT_NAME = "Test Project"
FAKE_PROJECT_DESCRIPTION = "Test Description"
DEFAULT_PROJECT_CONFIG = Config()
DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"
FAKE_LOG_ID = str(uuid.uuid4())


def test_project_validate_with_dict_response(
    mock_client_from_api_key: MagicMock,
    openai_chat_completion: "ChatCompletion",
    openai_messages_single_turn: list["ChatCompletionMessageParam"],
    openai_messages_conversational: list["ChatCompletionMessageParam"],
) -> None:
    expected_result = ProjectValidateResponse(
        is_bad_response=True,
        expert_answer=None,
        eval_scores={
            "response_helpfulness": EvalScores(
                score=0.8,
                triggered=True,
                triggered_escalation=False,
                triggered_guardrail=False,
            )
        },
        escalated_to_sme=True,
        should_guardrail=False,
        log_id=FAKE_LOG_ID,
    )
    mock_client_from_api_key.projects.validate.return_value = expected_result
    mock_client_from_api_key.projects.create.return_value.id = FAKE_PROJECT_ID
    mock_client_from_api_key.organization_id = FAKE_ORGANIZATION_ID
    project = Project.create(
        mock_client_from_api_key,
        FAKE_ORGANIZATION_ID,
        FAKE_PROJECT_NAME,
        FAKE_PROJECT_DESCRIPTION,
    )

    context = "Cities in France: Paris, Lyon, Marseille"
    query = "What is the capitol of France?"

    # single turn
    result = project.validate(
        messages=openai_messages_single_turn,
        response=openai_chat_completion,
        context=context,
        query=query,
    )

    assert result == expected_result
    mock_client_from_api_key.projects.validate.assert_called_once_with(
        FAKE_PROJECT_ID,
        messages=openai_messages_single_turn,
        response=openai_chat_completion,
        context=context,
        query=query,
        rewritten_question=None,
        custom_metadata=None,
        eval_scores=None,
        tools=None,
    )

    # conversational
    result = project.validate(
        messages=openai_messages_conversational,
        response=openai_chat_completion,
        context=context,
        query=query,
    )

    assert result == expected_result
    mock_client_from_api_key.projects.validate.assert_has_calls(
        [
            call(
                FAKE_PROJECT_ID,
                messages=openai_messages_single_turn,
                response=openai_chat_completion,
                context=context,
                query=query,
                rewritten_question=None,
                custom_metadata=None,
                eval_scores=None,
                tools=None,
            ),
            call(
                FAKE_PROJECT_ID,
                messages=openai_messages_conversational,
                response=openai_chat_completion,
                context=context,
                query=query,
                rewritten_question=None,
                custom_metadata=None,
                eval_scores=None,
                tools=None,
            ),
        ]
    )
    assert mock_client_from_api_key.projects.validate.call_count == 2


def test_project_validate_with_tools(
    mock_client_from_api_key: MagicMock,
    openai_chat_completion: "ChatCompletion",
    openai_messages_single_turn: list["ChatCompletionMessageParam"],
    openai_tools: list["ChatCompletionToolParam"],
) -> None:
    expected_result = ProjectValidateResponse(
        is_bad_response=True,
        expert_answer=None,
        eval_scores={
            "response_helpfulness": EvalScores(
                score=0.8,
                triggered=True,
                triggered_escalation=False,
                triggered_guardrail=False,
            )
        },
        escalated_to_sme=True,
        should_guardrail=False,
        log_id=FAKE_LOG_ID,
    )
    mock_client_from_api_key.projects.validate.return_value = expected_result
    mock_client_from_api_key.projects.create.return_value.id = FAKE_PROJECT_ID
    mock_client_from_api_key.organization_id = FAKE_ORGANIZATION_ID
    project = Project.create(
        mock_client_from_api_key,
        FAKE_ORGANIZATION_ID,
        FAKE_PROJECT_NAME,
        FAKE_PROJECT_DESCRIPTION,
    )

    context = "Cities in France: Paris, Lyon, Marseille"
    query = "What is the capitol of France?"

    # single turn
    result = project.validate(
        messages=openai_messages_single_turn,
        response=openai_chat_completion,
        tools=openai_tools,
        context=context,
        query=query,
    )

    assert result == expected_result
    mock_client_from_api_key.projects.validate.assert_called_once_with(
        FAKE_PROJECT_ID,
        messages=openai_messages_single_turn,
        response=openai_chat_completion,
        context=context,
        query=query,
        tools=openai_tools,
        rewritten_question=None,
        custom_metadata=None,
        eval_scores=None,
    )


def test_from_access_key(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.projects.access_keys.retrieve_project_id.return_value = (
        AccessKeyRetrieveProjectIDResponse(
            project_id=FAKE_PROJECT_ID,
        )
    )
    project = Project.from_access_key(DUMMY_ACCESS_KEY)
    assert project.id == FAKE_PROJECT_ID
    # should not call retrieve with an access key
    assert mock_client_from_access_key.projects.retrieve.call_count == 0
    assert mock_client_from_access_key.projects.access_keys.retrieve_project_id.call_count == 1


def test_from_access_key_missing_project(
    mock_client_from_access_key: MagicMock,
) -> None:
    """Test from_access_key when project_id is None"""
    mock_client_from_access_key.projects.access_keys.retrieve_project_id.side_effect = Exception("project ID not found")
    with pytest.raises(MissingProjectError):
        Project.from_access_key(DUMMY_ACCESS_KEY)


def test_create_project(mock_client_from_api_key: MagicMock, default_headers: dict[str, str]) -> None:
    """Test creating a new project"""
    mock_client_from_api_key.projects.create.return_value.id = FAKE_PROJECT_ID
    mock_client_from_api_key.organization_id = FAKE_ORGANIZATION_ID
    project = Project.create(
        mock_client_from_api_key,
        FAKE_ORGANIZATION_ID,
        FAKE_PROJECT_NAME,
        FAKE_PROJECT_DESCRIPTION,
    )
    mock_client_from_api_key.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=FAKE_PROJECT_DESCRIPTION,
        extra_headers=default_headers,
    )
    assert project.id == FAKE_PROJECT_ID
    assert mock_client_from_api_key.projects.retrieve.call_count == 0


def test_create_access_key(mock_client_from_api_key: MagicMock, default_headers: dict[str, str]) -> None:
    project = Project(mock_client_from_api_key, FAKE_PROJECT_ID)
    access_key_name = "Test Access Key"
    access_key_description = "Test Access Key Description"
    project.create_access_key(access_key_name, access_key_description)
    mock_client_from_api_key.projects.access_keys.create.assert_called_once_with(
        project_id=FAKE_PROJECT_ID,
        name=access_key_name,
        description=access_key_description,
        expires_at=None,
        extra_headers=default_headers,
    )


def test_create_access_key_no_access_key(
    mock_client_from_access_key: MagicMock,
) -> None:
    mock_error = Mock(response=Mock(status=401), body={"error": "Unauthorized"})

    mock_client_from_access_key.projects.access_keys.create.side_effect = AuthenticationError(
        "test", response=mock_error.response, body=mock_error.body
    )

    project = Project.from_access_key(DUMMY_ACCESS_KEY)

    with pytest.raises(AuthenticationError, match="See cleanlab_codex.Client.get_project"):
        project.create_access_key("test")


def test_init_nonexistent_project_id(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.projects.retrieve.return_value = None

    with pytest.raises(MissingProjectError):
        Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    assert mock_client_from_access_key.projects.retrieve.call_count == 1
