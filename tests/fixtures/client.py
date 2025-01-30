from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from beartype.typing import Generator
from codex import Codex as _Codex


@pytest.fixture
def mock_client() -> Generator[MagicMock, None, None]:
    with patch("cleanlab_codex.codex.init_codex_client") as mock_init:
        mock_client = MagicMock()
        mock_client.__class__ = _Codex  # type: ignore
        mock_init.return_value = mock_client
        yield mock_client
