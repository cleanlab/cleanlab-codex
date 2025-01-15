# ruff: noqa: DTZ005

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from codex.types.project_return_schema import Config, ProjectReturnSchema
from codex.types.users.myself.user_organizations_schema import UserOrganizationsSchema

from cleanlab_codex.codex import Codex
from cleanlab_codex.internal.project import MissingProjectIdError
from cleanlab_codex.types.entry import Entry, EntryCreate
from cleanlab_codex.types.organization import Organization
from cleanlab_codex.types.project import ProjectConfig

FAKE_PROJECT_ID = 1
FAKE_USER_ID = "Test User"
FAKE_ORGANIZATION_ID = "Test Organization"
FAKE_PROJECT_NAME = "Test Project"
FAKE_PROJECT_DESCRIPTION = "Test Description"
DEFAULT_PROJECT_CONFIG = ProjectConfig()


def test_list_organizations(mock_client: MagicMock):
    mock_client.users.myself.organizations.list.return_value = UserOrganizationsSchema(
        organizations=[
            Organization(
                organization_id=FAKE_ORGANIZATION_ID,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id=FAKE_USER_ID,
            )
        ],
    )
    codex = Codex("")
    organizations = codex.list_organizations()
    assert len(organizations) == 1
    assert organizations[0].organization_id == FAKE_ORGANIZATION_ID
    assert organizations[0].user_id == FAKE_USER_ID


def test_create_project(mock_client: MagicMock):
    mock_client.projects.create.return_value = ProjectReturnSchema(
        id=FAKE_PROJECT_ID,
        config=Config(),
        created_at=datetime.now(),
        created_by_user_id=FAKE_USER_ID,
        name=FAKE_PROJECT_NAME,
        organization_id=FAKE_ORGANIZATION_ID,
        updated_at=datetime.now(),
        description=FAKE_PROJECT_DESCRIPTION,
    )
    codex = Codex("")
    project_id = codex.create_project(
        FAKE_PROJECT_NAME, FAKE_ORGANIZATION_ID, FAKE_PROJECT_DESCRIPTION
    )
    mock_client.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=FAKE_PROJECT_DESCRIPTION,
    )
    assert project_id == FAKE_PROJECT_ID


def test_add_entries(mock_client: MagicMock):
    answered_entry_create = EntryCreate(
        question="What is the capital of France?",
        answer="Paris",
    )
    unanswered_entry_create = EntryCreate(
        question="What is the capital of Germany?",
    )
    codex = Codex("")
    codex.add_entries(
        [answered_entry_create, unanswered_entry_create], project_id=FAKE_PROJECT_ID
    )

    for call, entry in zip(
        mock_client.projects.entries.create.call_args_list,
        [answered_entry_create, unanswered_entry_create],
    ):
        assert call.args[0] == FAKE_PROJECT_ID
        assert call.kwargs["question"] == entry["question"]
        assert call.kwargs["answer"] == entry.get("answer")


def test_create_project_access_key(mock_client: MagicMock):
    codex = Codex("")
    access_key_name = "Test Access Key"
    access_key_description = "Test Access Key Description"
    codex.create_project_access_key(
        FAKE_PROJECT_ID, access_key_name, access_key_description
    )
    mock_client.projects.access_keys.create.assert_called_once_with(
        project_id=FAKE_PROJECT_ID,
        name=access_key_name,
        description=access_key_description,
    )


def test_query_no_project_id(mock_client: MagicMock):
    mock_client.access_key = None
    codex = Codex("")

    with pytest.raises(MissingProjectIdError):
        codex.query("What is the capital of France?")


def test_query_read_only(mock_client: MagicMock):
    mock_client.access_key = None
    mock_client.projects.entries.query.return_value = None

    codex = Codex("")
    res = codex.query(
        "What is the capital of France?", read_only=True, project_id=FAKE_PROJECT_ID
    )
    mock_client.projects.entries.query.assert_called_once_with(
        FAKE_PROJECT_ID, question="What is the capital of France?"
    )
    mock_client.projects.entries.add_question.assert_not_called()
    assert res == (None, None)


def test_query_question_found_fallback_answer(mock_client: MagicMock):
    unanswered_entry = Entry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        question="What is the capital of France?",
        answer=None,
    )
    mock_client.projects.entries.query.return_value = unanswered_entry
    codex = Codex("")
    res = codex.query("What is the capital of France?", project_id=FAKE_PROJECT_ID)
    assert res == (None, unanswered_entry)


def test_query_question_not_found_fallback_answer(mock_client: MagicMock):
    mock_client.projects.entries.query.return_value = None
    mock_client.projects.entries.add_question.return_value = None

    codex = Codex("")
    res = codex.query("What is the capital of France?", fallback_answer="Paris")
    assert res == ("Paris", None)


def test_query_answer_found(mock_client: MagicMock):
    answered_entry = Entry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        question="What is the capital of France?",
        answer="Paris",
    )
    mock_client.projects.entries.query.return_value = answered_entry
    codex = Codex("")
    res = codex.query("What is the capital of France?", project_id=FAKE_PROJECT_ID)
    assert res == ("Paris", answered_entry)
