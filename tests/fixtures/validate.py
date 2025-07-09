from typing import cast

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


@pytest.fixture
def openai_chat_completion() -> ChatCompletion:
    """Fixture that returns a static fake OpenAI ChatCompletion object."""
    raw_response = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1719876543,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Paris",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
    }

    return ChatCompletion.model_validate(raw_response)


@pytest.fixture
def openai_messages_single_turn() -> list[ChatCompletionMessageParam]:
    """Fixture that returns a single-turn message format."""
    return [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "What is the capital of France?"})]


@pytest.fixture
def openai_messages_bad_no_user() -> list[ChatCompletionMessageParam]:
    """Fixture that returns invalid messages (missing required user message)."""
    return [
        cast(ChatCompletionAssistantMessageParam, {"role": "assistant", "content": "hi"}),
        cast(ChatCompletionSystemMessageParam, {"role": "system", "content": "sys"}),
    ]


@pytest.fixture
def openai_messages_conversational() -> list[ChatCompletionMessageParam]:
    """Fixture that returns a conversational message format."""
    return [
        cast(ChatCompletionSystemMessageParam, {"role": "system", "content": "You are a helpful assistant."}),
        cast(ChatCompletionUserMessageParam, {"role": "user", "content": "I love France!"}),
        cast(ChatCompletionAssistantMessageParam, {"role": "assistant", "content": "That's great!"}),
        cast(ChatCompletionUserMessageParam, {"role": "user", "content": "What is its capital?"}),
    ]
