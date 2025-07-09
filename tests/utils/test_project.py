from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from cleanlab_codex.utils.project import verify_messages_format, verify_response_format
from tests.fixtures.validate import (
    openai_chat_completion,
    openai_messages_bad_no_user,
    openai_messages_conversational,
    openai_messages_single_turn,
)

assert openai_chat_completion is not None  # needed as dummy so hatch does not delete
assert openai_messages_single_turn is not None  # needed as dummy so hatch does not delete
assert openai_messages_conversational is not None  # needed as dummy so hatch does not delete
assert openai_messages_bad_no_user is not None  # needed as dummy so hatch does not delete


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


def test_raises_if_no_user_message(openai_messages_bad_no_user: list["ChatCompletionMessageParam"]) -> None:
    with pytest.raises(ValueError, match="At least one user message is required"):
        verify_messages_format(openai_messages_bad_no_user)


def test_valid_response_string() -> None:
    verify_response_format("this is a plain string")  # Should not raise


def test_valid_response_dict(openai_chat_completion: "ChatCompletion") -> None:
    verify_response_format(openai_chat_completion)  # Should not raise


def test_raises_on_empty_content(openai_chat_completion: "ChatCompletion") -> None:
    openai_chat_completion.choices[0].message.content = ""  # Set content to empty string
    with pytest.raises(ValueError, match="Response message content must be a non-empty string."):
        verify_response_format(openai_chat_completion)
