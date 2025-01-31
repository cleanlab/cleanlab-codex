from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_client_from_access_key() -> Generator[MagicMock, None, None]:
    with patch("cleanlab_codex.project.client_from_access_key") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_client_from_api_key() -> Generator[MagicMock, None, None]:
    with patch("cleanlab_codex.client.client_from_api_key") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        yield mock_client
