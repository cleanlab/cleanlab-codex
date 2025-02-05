from __future__ import annotations

import os
import re

from codex import Codex as _Codex

ACCESS_KEY_PATTERN = r"^sk-.*-.*$"


class MissingAuthKeyError(ValueError):
    """Raised when no API key or access key is provided."""

    def __str__(self) -> str:
        return "No API key or access key provided"


def is_access_key(key: str) -> bool:
    return re.match(ACCESS_KEY_PATTERN, key) is not None


def client_from_api_key(key: str | None = None) -> _Codex:
    """
    Initialize a Codex SDK client using a user-level API key.

    Args:
        key (str | None): The API key to use to authenticate the client. If not provided, the client will be authenticated using the `CODEX_API_KEY` environment variable.

    Returns:
        _Codex: The initialized Codex client.
    """
    if not (key := key or os.getenv("CODEX_API_KEY")):
        raise MissingAuthKeyError

    client = _Codex(api_key=key)
    client.users.myself.api_key.retrieve()  # check if the api key is valid
    return client


def client_from_access_key(key: str | None = None) -> _Codex:
    """
    Initialize a Codex SDK client using a project-level access key.

    Args:
        key (str | None): The access key to use to authenticate the client. If not provided, the client will be authenticated using the `CODEX_ACCESS_KEY` environment variable.

    Returns:
        _Codex: The initialized Codex client.
    """
    if not (key := key or os.getenv("CODEX_ACCESS_KEY")):
        raise MissingAuthKeyError

    return _Codex(access_key=key)
