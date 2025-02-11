from unittest.mock import MagicMock

import pytest

from cleanlab_codex.codex_backup import CodexBackup

MOCK_BACKUP_RESPONSE = "This is a test response"
FALLBACK_MESSAGE = "Based on the available information, I cannot provide a complete answer to this question."
TEST_MESSAGE = "Hello, world!"


class MockApp:
    def chat(self, user_message: str) -> str:
        # Just echo the user message
        return user_message

@pytest.fixture
def mock_app() -> MockApp:
    return MockApp()


def test_codex_backup(mock_app: MockApp) -> None:
    # Create a mock project directly
    mock_project = MagicMock()
    mock_project.query.return_value = (MOCK_BACKUP_RESPONSE,)

    # Echo works well
    query = TEST_MESSAGE
    response = mock_app.chat(query)
    assert response == query

    # Backup works well for fallback responses
    codex_backup = CodexBackup.from_project(mock_project)
    query = FALLBACK_MESSAGE
    response = mock_app.chat(query)
    assert response == query
    response = codex_backup.run(response, query=query)
    assert response == MOCK_BACKUP_RESPONSE, f"Response was {response}"


def test_backup_handler(mock_app: MockApp) -> None:
    mock_project = MagicMock()
    mock_project.query.return_value = (MOCK_BACKUP_RESPONSE,)

    mock_handler = MagicMock()
    mock_handler.return_value = None

    codex_backup = CodexBackup.from_project(mock_project, primary_system=mock_app, backup_handler=mock_handler)

    query = TEST_MESSAGE
    response = mock_app.chat(query)
    assert response == query

    response = codex_backup.run(response, query=query)
    assert response == query, f"Response was {response}"

    # Handler should not be called for good responses
    assert mock_handler.call_count == 0

    query = FALLBACK_MESSAGE
    response = mock_app.chat(query)
    assert response == query
    response = codex_backup.run(response, query=query)
    assert response == MOCK_BACKUP_RESPONSE, f"Response was {response}"

    # Handler should be called for bad responses
    assert mock_handler.call_count == 1
    # The MockApp is the second argument to the handler, i.e. it has the necessary context
    # to handle the new response
    assert mock_handler.call_args.kwargs["primary_system"] == mock_app
