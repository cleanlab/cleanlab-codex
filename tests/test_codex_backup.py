import pytest
from unittest.mock import MagicMock

from cleanlab_codex.codex_backup import CodexBackup

# TODO: Remove this skip once we update codex_backup.py
pytest.skip(allow_module_level=True)


MOCK_BACKUP_RESPONSE = "This is a test response"
FALLBACK_MESSAGE = "Based on the available information, I cannot provide a complete answer to this question."
TEST_MESSAGE = "Hello, world!"


def test_codex_backup(mock_client: MagicMock):
    mock_response = MagicMock()
    mock_response.answer = MOCK_BACKUP_RESPONSE
    mock_client.projects.entries.query.return_value = mock_response

    codex_backup = CodexBackup.from_access_key("")

    class MockApp:
        @codex_backup.to_decorator()
        def chat(self, user_message: str) -> str:
            # Just echo the user message
            return user_message

    app = MockApp()

    # Echo works well
    response = app.chat(TEST_MESSAGE)
    assert response == TEST_MESSAGE

    # Backup works well for fallback responses
    response = app.chat(FALLBACK_MESSAGE)
    assert response == MOCK_BACKUP_RESPONSE


def test_backup_handler(mock_client: MagicMock):
    mock_response = MagicMock()
    mock_response.answer = MOCK_BACKUP_RESPONSE
    mock_client.projects.entries.query.return_value = mock_response

    mock_handler = MagicMock()
    mock_handler.return_value = None
    codex_backup = CodexBackup.from_access_key("", backup_handler=mock_handler)

    class MockApp:
        @codex_backup.to_decorator()
        def chat(self, user_message: str) -> str:
            # Just echo the user message
            return user_message

    app = MockApp()

    response = app.chat(TEST_MESSAGE)
    assert response == TEST_MESSAGE

    # Handler should not be called for good responses
    assert mock_handler.call_count == 0

    response = app.chat(FALLBACK_MESSAGE)
    assert response == MOCK_BACKUP_RESPONSE

    # Handler should be called for bad responses
    assert mock_handler.call_count == 1
    # The MockApp is the second argument to the handler, i.e. it has the necessary context
    # to handle the new response
    assert mock_handler.call_args.kwargs["decorated_instance"] == app
