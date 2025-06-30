from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from cleanlab_tlm.utils.chat import (
    ASSISTANT_ROLE,
    SYSTEM_ROLES,
    TOOL_ROLE,
    USER_ROLE,
)

VALID_MESSAGE_ROLES: list[str] = [
    *SYSTEM_ROLES,
    USER_ROLE,
    TOOL_ROLE,
    ASSISTANT_ROLE,
]  # TODO: possibly move this to TLM?


def verify_messages_format(
    messages: list["ChatCompletionMessageParam"],
) -> None:
    """Check if the messages are in the correct format for OpenAI chat completions.

    Args:
        messages: List of chat messages to validate.

    Raises:
        TypeError: If any message is not a dictionary.
        ValueError: If any message does not have 'role' or 'content' keys.
    """
    user_message_found = False

    if len(messages) == 0:
        msg = "Messages list cannot be empty."
        raise ValueError(msg)

    for message in messages:
        if not isinstance(message, dict):
            msg = f"Each message must be a dictionary, got {type(message)}"
            raise TypeError(msg)
        if "role" not in message or "content" not in message:
            msg = "Each message must contain 'role' and 'content' keys."
            raise ValueError(msg)
        if not isinstance(message["role"], str):
            msg = f"Message role must be a string, got {type(message['role'])}"
            raise TypeError(msg)
        if not isinstance(message["content"], str):
            msg = f"Message content must be a string, got {type(message['content'])}"
            raise TypeError(msg)
        if message["role"] not in VALID_MESSAGE_ROLES:
            valid_roles_str = ", ".join(VALID_MESSAGE_ROLES)
            msg = f"Invalid message role '{message['role']}'. Valid roles are: {valid_roles_str}"
            raise ValueError(msg)
        if message["role"] == USER_ROLE:
            user_message_found = True

    if not user_message_found:
        msg = f"At least one user message is required in the messages list under role {USER_ROLE}."
        raise ValueError(msg)


def get_latest_prompt_from_messages(
    messages: list["ChatCompletionMessageParam"],
) -> str:
    """Extract the latest user message from a list of chat messages."""
    for message in reversed(messages):
        if message["role"] == USER_ROLE:
            return message["content"]
    msg = "No user message found in the messages list."
    raise ValueError(msg)
