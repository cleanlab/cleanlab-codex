import re

from codex import Codex as _Codex

ACCESS_KEY_PATTERN = r"^sk-.*-.*$"


def is_access_key(key: str) -> bool:
    return re.match(ACCESS_KEY_PATTERN, key) is not None


def init_codex_client(key: str) -> _Codex:
    if is_access_key(key):
        return _Codex(access_key=key)

    return _Codex(api_key=key)
