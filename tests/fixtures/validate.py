import pytest
from cleanlab_tlm.utils.chat import _ASSISTANT_ROLE, _SYSTEM_ROLES, _USER_ROLE
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


@pytest.fixture
def openai_chat_completion() -> "ChatCompletion":
    """Fixture that returns a static fake OpenAI ChatCompletion object."""
    return ChatCompletion(
        id="chatcmpl-test123",
        object="chat.completion",
        created=1719876543,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": f"{_ASSISTANT_ROLE}",
                    "content": "Paris",
                },
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
    )


@pytest.fixture
def openai_messages_single_turn() -> list["ChatCompletionMessageParam"]:
    """Fixture that returns a single-turn message format."""
    return [{"role": f"{_USER_ROLE}", "content": "What is the capital of France?"}]


@pytest.fixture
def openai_messages_conversational() -> list["ChatCompletionMessageParam"]:
    """Fixture that returns a conversational message format."""
    return [
        {"role": f"{_SYSTEM_ROLES[0]}", "content": "You are a helpful assistant."},
        {"role": f"{_USER_ROLE}", "content": "I love France!"},
        {"role": f"{_ASSISTANT_ROLE}", "content": "That's great!"},
        {"role": f"{_USER_ROLE}", "content": "What is its capital?"},
    ]
