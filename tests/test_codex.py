from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from codex import Codex as _Codex

from cleanlab_codex.codex import Codex

fake_project_id = 1


@pytest.fixture
def mock_client() -> Generator[_Codex, None, None]:
    with patch("cleanlab_codex.codex.init_codex_client", return_value=MagicMock()) as mock:
        yield mock


def test_query_read_only(mock_client: _Codex):
    mock_client.projects.entries.query.return_value = None  # type: ignore
    codex = Codex("")
    res = codex.query("What is the capital of France?", read_only=True, project_id=fake_project_id)
    mock_client.projects.entries.query.assert_called_once_with(  # type: ignore
        fake_project_id, "What is the capital of France?"
    )
    mock_client.projects.entries.add_question.assert_not_called()  # type: ignore
    assert res == (None, None)
