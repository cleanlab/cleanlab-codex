from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from codex import Codex as _Codex

from cleanlab_codex.internal.utils import MissingAuthKeyError, init_codex_client, is_access_key

DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"
DUMMY_API_KEY = "GP0FzPfA7wYy5L64luII2YaRT2JoSXkae7WEo7dH6Bw"


def test_is_access_key() -> None:
    assert is_access_key(DUMMY_ACCESS_KEY)
    assert not is_access_key(DUMMY_API_KEY)


def test_init_codex_client_access_key() -> None:
    mock_client = MagicMock()
    mock_client.__class__ = _Codex  # type: ignore
    with patch("cleanlab_codex.internal.utils._Codex", autospec=True, return_value=mock_client) as mock_init:
        mock_client.projects.access_keys.retrieve_project_id.return_value = "test_project_id"
        client = init_codex_client(DUMMY_ACCESS_KEY)
        mock_init.assert_called_once_with(access_key=DUMMY_ACCESS_KEY)
        assert client is not None


def test_init_codex_client_api_key() -> None:
    mock_client = MagicMock()
    mock_client.__class__ = _Codex  # type: ignore
    with patch("cleanlab_codex.internal.utils._Codex", autospec=True, return_value=mock_client) as mock_init:
        mock_client.users.myself.api_key.retrieve.return_value = "test_project_id"
        client = init_codex_client(DUMMY_API_KEY)
        mock_init.assert_called_once_with(api_key=DUMMY_API_KEY)
        assert client is not None


def test_init_codex_client_no_key() -> None:
    with pytest.raises(MissingAuthKeyError):
        init_codex_client()


def test_init_codex_client_access_key_env_var() -> None:
    with patch.dict(os.environ, {"CODEX_ACCESS_KEY": DUMMY_ACCESS_KEY}):
        mock_client = MagicMock()
        mock_client.__class__ = _Codex  # type: ignore
        with patch(
            "cleanlab_codex.internal.utils._Codex",
            autospec=True,
            return_value=mock_client,
        ) as mock_init:
            mock_client.projects.access_keys.retrieve_project_id.return_value = "test_project_id"
            client = init_codex_client()
            mock_init.assert_called_once_with(access_key=DUMMY_ACCESS_KEY)
            assert client is not None


def test_init_codex_client_api_key_env_var() -> None:
    with patch.dict(os.environ, {"CODEX_API_KEY": DUMMY_API_KEY}):
        mock_client = MagicMock()
        mock_client.__class__ = _Codex  # type: ignore
        with patch(
            "cleanlab_codex.internal.utils._Codex",
            autospec=True,
            return_value=mock_client,
        ) as mock_init:
            mock_client.users.myself.api_key.retrieve.return_value = "test_project_id"
            client = init_codex_client()
            mock_init.assert_called_once_with(api_key=DUMMY_API_KEY)
            assert client is not None
