import re

from codex import Codex as _Codex

ACCESS_KEY_PATTERN = r"^sk-.*-.*$"


def is_access_key(key: str) -> bool:
    return re.match(ACCESS_KEY_PATTERN, key) is not None


def init_codex_client(key: str) -> _Codex:
    if is_access_key(key):
        client = _Codex(access_key=key)
        client.projects.access_keys.retrieve_project_id()  # check if the access key is valid
        return client

    client = _Codex(api_key=key)
    client.users.myself.api_key.retrieve()  # check if the api key is valid
    return client
