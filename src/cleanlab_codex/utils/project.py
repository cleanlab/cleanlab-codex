from typing import Any

from cleanlab_tlm.utils.chat import (
    _ASSISTANT_ROLE,
    _SYSTEM_ROLES,
    _TOOL_ROLE,
    _USER_ROLE,
)

VALID_MESSAGE_ROLES: list[str] = [
    *_SYSTEM_ROLES,
    _USER_ROLE,
    _TOOL_ROLE,
    _ASSISTANT_ROLE,
]  # TODO: possibly move this to TLM?


def verify_messages_format(
    messages: Any,
) -> None:
    """Check if the messages are in the correct format for OpenAI chat completions.

    Args:
        messages: List of chat messages to validate.

    Raises:
        TypeError: If any message is not a dictionary.
        ValueError: If any message does not have 'role' or 'content' keys.
    """
    user_message_found = False

    if not isinstance(messages, list):
        msg = f"Messages must be a list of dictionaries, got {type(messages)}"
        raise TypeError(msg)
    if len(messages) == 0:
        msg = "Messages list cannot be empty. At least one message is required."
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
        if message["role"] == _USER_ROLE:
            user_message_found = True

    if not user_message_found:
        msg = f"At least one user message is required in the messages list under role {_USER_ROLE}."
        raise ValueError(msg)


def verify_response_format(response: Any) -> None:
    """Check if the response is in the correct format for OpenAI chat completions or string.

    Args:
        response: The response to validate.

    Raises:
        TypeError: If the response is not a ChatCompletion-like object or string.
        ValueError: If the response does not have valid choices or content.
    """
    if isinstance(response, str):
        return

    try:
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            msg = "Response message content must be a non-empty string."
            raise ValueError(msg)
    except (AttributeError, IndexError, TypeError) as e:
        msg = f"Response must be a string or ChatCompletion-like object. Got {type(response)}"
        raise TypeError(msg) from e
