from typing import TYPE_CHECKING

import pytest
from cleanlab_tlm.utils.chat import _ASSISTANT_ROLE, _SYSTEM_ROLES, _USER_ROLE

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from cleanlab_codex.utils.project import verify_messages_format, verify_response_format
from tests.fixtures.validate import openai_chat_completion, openai_messages_conversational, openai_messages_single_turn

assert openai_chat_completion is not None  # needed as dummy so hatch does not delete
assert openai_messages_single_turn is not None  # needed as dummy so hatch does not delete
assert openai_messages_conversational is not None  # needed as dummy so hatch does not delete


def test_valid_single_user_message(openai_messages_single_turn: list["ChatCompletionMessageParam"]) -> None:
    verify_messages_format(openai_messages_single_turn)  # Should not raise


def test_valid_multiple_roles(openai_messages_conversational: list["ChatCompletionMessageParam"]) -> None:
    verify_messages_format(openai_messages_conversational)  # Should not raise


def test_raises_on_non_list() -> None:
    with pytest.raises(TypeError, match="Messages must be a list"):
        verify_messages_format("not a list")  # type: ignore[arg-type]


def test_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        verify_messages_format([])


def test_raises_on_non_dict_message() -> None:
    with pytest.raises(TypeError, match="Each message must be a dictionary"):
        verify_messages_format(["not a dict"])


def test_raises_on_missing_keys() -> None:
    with pytest.raises(ValueError, match="must contain 'role' and 'content'"):
        verify_messages_format([{"role": _USER_ROLE}])


def test_raises_on_invalid_role_type() -> None:
    with pytest.raises(TypeError, match="Message role must be a string"):
        verify_messages_format([{"role": 123, "content": "text"}])


def test_raises_on_invalid_content_type() -> None:
    with pytest.raises(TypeError, match="Message content must be a string"):
        verify_messages_format([{"role": _USER_ROLE, "content": 123}])


def test_raises_on_invalid_role_value() -> None:
    with pytest.raises(ValueError, match="Invalid message role"):
        verify_messages_format([{"role": "hacker", "content": "intrude"}])


def test_raises_if_no_user_message() -> None:
    messages = [
        {"role": _ASSISTANT_ROLE, "content": "hi"},
        {"role": _SYSTEM_ROLES[0], "content": "sys"},
    ]
    with pytest.raises(ValueError, match="At least one user message is required"):
        verify_messages_format(messages)


def test_valid_response_string() -> None:
    verify_response_format("this is a plain string")  # Should not raise


def test_valid_response_dict(openai_chat_completion: "ChatCompletion") -> None:
    verify_response_format(openai_chat_completion)  # Should not raise


def test_raises_on_non_dict_or_string() -> None:
    with pytest.raises(TypeError, match="Response must be a string or ChatCompletion-like object. Got <class 'int'>"):
        verify_response_format(123)


def test_raises_on_empty_content(openai_chat_completion: "ChatCompletion") -> None:
    openai_chat_completion.choices[0].message.content = ""  # Set content to empty string
    with pytest.raises(ValueError, match="Response message content must be a non-empty string."):
        verify_response_format(openai_chat_completion)
