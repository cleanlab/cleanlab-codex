from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_client() -> Generator[MagicMock, None, None]:
    with patch("cleanlab_codex.codex.init_codex_client") as mock_init:
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        yield mock_client
