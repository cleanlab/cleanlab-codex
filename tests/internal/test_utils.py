from unittest.mock import patch

from cleanlab_codex.internal.utils import init_codex_client, is_access_key

DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"
DUMMY_API_KEY = "GP0FzPfA7wYy5L64luII2YaRT2JoSXkae7WEo7dH6Bw"


def test_is_access_key() -> None:
    assert is_access_key(DUMMY_ACCESS_KEY)
    assert not is_access_key(DUMMY_API_KEY)


def test_init_codex_client_access_key() -> None:
    with patch("cleanlab_codex.internal.utils._Codex", autospec=True) as mock_codex:
        client = init_codex_client(DUMMY_ACCESS_KEY)
        mock_codex.assert_called_once_with(access_key=DUMMY_ACCESS_KEY)
        assert client is not None


def test_init_codex_client_api_key() -> None:
    with patch("cleanlab_codex.internal.utils._Codex", autospec=True) as mock_codex:
        client = init_codex_client(DUMMY_API_KEY)
        mock_codex.assert_called_once_with(api_key=DUMMY_API_KEY)
        assert client is not None
