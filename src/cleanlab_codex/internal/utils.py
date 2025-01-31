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


def init_codex_client(key: str | None = None) -> _Codex:
    """
    Initialize a Codex SDK client using an API key or access key.

    Args:
        key (str | None): The API key or access key to use to authenticate the client. If not provided, the client will be authenticated using the `CODEX_API_KEY` or `CODEX_ACCESS_KEY` environment variables.

    Returns:
        _Codex: The initialized Codex client.
    """
    if key is None:
        if api_key := os.getenv("CODEX_API_KEY"):
            return client_from_api_key(api_key)
        if access_key := os.getenv("CODEX_ACCESS_KEY"):
            return client_from_access_key(access_key)

        raise MissingAuthKeyError

    if is_access_key(key):
        return client_from_access_key(key)

    return client_from_api_key(key)


def client_from_api_key(key: str) -> _Codex:
    client = _Codex(api_key=key)
    client.users.myself.api_key.retrieve()  # check if the api key is valid
    return client


def client_from_access_key(key: str) -> _Codex:
    client = _Codex(access_key=key)
    client.projects.access_keys.retrieve_project_id()  # check if the access key is valid
    return client
