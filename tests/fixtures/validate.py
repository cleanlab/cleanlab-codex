from typing import List, Literal, Optional

import pytest
from pydantic import BaseModel


class ChatCompletionMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str]


class Choice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None


@pytest.fixture
def mock_openai_chat_completion() -> MockChatCompletion:
    return MockChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="Paris"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )
