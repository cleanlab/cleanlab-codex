from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from cleanlab_codex.internal.analytics import _AnalyticsMetadata


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


@pytest.fixture
def default_headers() -> dict[str, str]:
    return _AnalyticsMetadata().to_headers()
