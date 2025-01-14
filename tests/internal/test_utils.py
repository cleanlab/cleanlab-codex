from cleanlab_codex.internal.utils import is_access_key

DUMMY_ACCESS_KEY = "sk-1-EMOh6UrRo7exTEbEi8_azzACAEdtNiib2LLa1IGo6kA"
DUMMY_API_KEY = "GP0FzPfA7wYy5L64luII2YaRT2JoSXkae7WEo7dH6Bw"


def test_is_access_key():
    assert is_access_key(DUMMY_ACCESS_KEY)
    assert not is_access_key(DUMMY_API_KEY)
