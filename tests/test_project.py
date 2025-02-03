import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import pytest
from codex import AuthenticationError
from codex.types.project_create_params import Config
from codex.types.projects.access_key_retrieve_project_id_response import AccessKeyRetrieveProjectIDResponse

from cleanlab_codex.project import MissingProjectError, Project
from cleanlab_codex.types.entry import Entry, EntryCreate

FAKE_PROJECT_ID = str(uuid.uuid4())
FAKE_USER_ID = "Test User"
FAKE_ORGANIZATION_ID = "Test Organization"
FAKE_PROJECT_NAME = "Test Project"
FAKE_PROJECT_DESCRIPTION = "Test Description"
DEFAULT_PROJECT_CONFIG = Config()
DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"


def test_from_access_key(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.projects.access_keys.retrieve_project_id.return_value = (
        AccessKeyRetrieveProjectIDResponse(
            project_id=FAKE_PROJECT_ID,
        )
    )
    project = Project.from_access_key(DUMMY_ACCESS_KEY)
    assert project.project_id == FAKE_PROJECT_ID


def test_from_access_key_missing_project(mock_client_from_access_key: MagicMock) -> None:
    """Test from_access_key when project_id is None"""
    mock_client_from_access_key.projects.access_keys.retrieve_project_id.side_effect = Exception("project ID not found")
    with pytest.raises(MissingProjectError):
        Project.from_access_key(DUMMY_ACCESS_KEY)


def test_create_project(mock_client_from_access_key: MagicMock) -> None:
    """Test creating a new project"""
    mock_client_from_access_key.projects.create.return_value.id = FAKE_PROJECT_ID
    mock_client_from_access_key.organization_id = FAKE_ORGANIZATION_ID
    project = Project.create(
        mock_client_from_access_key, FAKE_ORGANIZATION_ID, FAKE_PROJECT_NAME, FAKE_PROJECT_DESCRIPTION
    )
    mock_client_from_access_key.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=FAKE_PROJECT_DESCRIPTION,
    )
    assert project.project_id == FAKE_PROJECT_ID


def test_add_entries(mock_client_from_access_key: MagicMock) -> None:
    answered_entry_create = EntryCreate(
        question="What is the capital of France?",
        answer="Paris",
    )
    unanswered_entry_create = EntryCreate(
        question="What is the capital of Germany?",
    )
    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    project.add_entries([answered_entry_create, unanswered_entry_create])

    for call, entry in zip(
        mock_client_from_access_key.projects.entries.create.call_args_list,
        [answered_entry_create, unanswered_entry_create],
    ):
        assert call.args[0] == FAKE_PROJECT_ID
        assert call.kwargs["question"] == entry["question"]
        assert call.kwargs["answer"] == entry.get("answer")


def test_add_entries_no_access_key(mock_client_from_access_key: MagicMock) -> None:
    mock_error = Mock(response=Mock(status=401), body={"error": "Unauthorized"})

    mock_client_from_access_key.projects.entries.create.side_effect = AuthenticationError(
        "test", response=mock_error.response, body=mock_error.body
    )

    answered_entry_create = EntryCreate(
        question="What is the capital of France?",
        answer="Paris",
    )

    project = Project.from_access_key(DUMMY_ACCESS_KEY)

    with pytest.raises(AuthenticationError, match="See cleanlab_codex.Client.get_project"):
        project.add_entries([answered_entry_create])


def test_create_access_key(mock_client_from_access_key: MagicMock) -> None:
    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    access_key_name = "Test Access Key"
    access_key_description = "Test Access Key Description"
    project.create_access_key(access_key_name, access_key_description)
    mock_client_from_access_key.projects.access_keys.create.assert_called_once_with(
        project_id=FAKE_PROJECT_ID,
        name=access_key_name,
        description=access_key_description,
        expires_at=None,
    )


def test_create_access_key_no_access_key(mock_client_from_access_key: MagicMock) -> None:
    mock_error = Mock(response=Mock(status=401), body={"error": "Unauthorized"})

    mock_client_from_access_key.projects.access_keys.create.side_effect = AuthenticationError(
        "test", response=mock_error.response, body=mock_error.body
    )

    project = Project.from_access_key(DUMMY_ACCESS_KEY)

    with pytest.raises(AuthenticationError, match="See cleanlab_codex.Client.get_project"):
        project.create_access_key("test")


def test_create_nonexistent_project_id(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.projects.retrieve.return_value = None

    with pytest.raises(MissingProjectError):
        Project(mock_client_from_access_key, FAKE_PROJECT_ID)


def test_query_read_only(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.access_key = None
    mock_client_from_access_key.projects.entries.query.return_value = None

    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    res = project.query("What is the capital of France?", read_only=True)
    mock_client_from_access_key.projects.entries.query.assert_called_once_with(
        FAKE_PROJECT_ID, question="What is the capital of France?"
    )
    mock_client_from_access_key.projects.entries.add_question.assert_not_called()
    assert res == (None, None)


def test_query_question_found_fallback_answer(mock_client_from_access_key: MagicMock) -> None:
    unanswered_entry = Entry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
        question="What is the capital of France?",
        answer=None,
    )
    mock_client_from_access_key.projects.entries.query.return_value = unanswered_entry
    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    res = project.query("What is the capital of France?")
    assert res == (None, unanswered_entry)


def test_query_question_not_found_fallback_answer(mock_client_from_access_key: MagicMock) -> None:
    mock_client_from_access_key.projects.entries.query.return_value = None
    mock_client_from_access_key.projects.entries.add_question.return_value = None

    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    res = project.query("What is the capital of France?", fallback_answer="Paris")
    assert res == ("Paris", None)


def test_query_add_question_when_not_found(mock_client_from_access_key: MagicMock) -> None:
    """Test that query adds question when not found and not read_only"""
    mock_client_from_access_key.projects.entries.query.return_value = None
    new_entry = Entry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
        question="What is the capital of France?",
        answer=None,
    )
    mock_client_from_access_key.projects.entries.add_question.return_value = new_entry

    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    res = project.query("What is the capital of France?")

    mock_client_from_access_key.projects.entries.add_question.assert_called_once_with(
        FAKE_PROJECT_ID, question="What is the capital of France?"
    )
    assert res == (None, new_entry)


def test_query_answer_found(mock_client_from_access_key: MagicMock) -> None:
    answered_entry = Entry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
        question="What is the capital of France?",
        answer="Paris",
    )
    mock_client_from_access_key.projects.entries.query.return_value = answered_entry
    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    res = project.query("What is the capital of France?")
    assert res == ("Paris", answered_entry)


def test_add_entries_empty_list(mock_client_from_access_key: MagicMock) -> None:
    """Test adding an empty list of entries"""
    project = Project(mock_client_from_access_key, FAKE_PROJECT_ID)
    project.add_entries([])
    mock_client_from_access_key.projects.entries.create.assert_not_called()
