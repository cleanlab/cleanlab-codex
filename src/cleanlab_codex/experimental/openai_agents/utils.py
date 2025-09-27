"""Utilities for formatting responses from OpenAI's Responses API and tools from chat messages.

This module provides helper functions for working with responses from OpenAI's new Responses API format,
which returns responses as lists containing different types of output elements (messages, function calls, etc.).

It also includes a utility to extract tool results from chat messages based on tool names.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, cast

if TYPE_CHECKING:
    from agents.items import ModelResponse
    from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam

# Import constants from the main chat module
from cleanlab_tlm.utils.chat import (
    _TOOL_CALL_TAG_END,
    _TOOL_CALL_TAG_START,
    _TOOL_RESPONSE_TAG_END,
    _TOOL_RESPONSE_TAG_START,
)
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam


def form_response_string_responses_api_list(response_list: List[Dict[str, Any]]) -> str:
    """
    Form a single string representing the response from OpenAI's new Responses API format.

    The new Responses API returns a list of different types of response elements, where each element
    can be a message, function_call, or function_call_output with fields like:
    - 'arguments': JSON string containing function arguments
    - 'call_id': Unique identifier for the function call
    - 'name': Function name
    - 'type': Type of response ('function_call', 'message', etc.)
    - 'id': Unique identifier for the response element
    - 'status': Status of the response ('completed', etc.)

    Args:
        response_list (List[Dict[str, Any]]): A list of response elements from the Responses API.
            Each element should contain fields like 'type', 'call_id', 'name', 'arguments', 'status', etc.
            Example format:
            [
                {
                    'arguments': '{"a":5,"b":3,"operation":"add"}',
                    'call_id': 'call_poNImPOog6UoxS5wwLXyWvld',
                    'name': 'safe_calculation',
                    'type': 'function_call',
                    'id': 'fc_68d58db9443881909f2769d30776355e02be88949bbce8bf',
                    'status': 'completed'
                }
            ]

    Returns:
        str: A formatted string containing the response content and any function calls.
             Function calls are formatted as XML tags containing JSON with function
             name and arguments, consistent with the format used in other chat utilities.
    """
    content_parts = []

    for element in response_list:
        element_type = element.get("type", "")

        if element_type == "message":
            # Handle message content
            content = element.get("content", "")
            if isinstance(content, list):
                # Handle content as list of content parts
                text_parts = []
                for content_part in content:
                    if isinstance(content_part, dict) and content_part.get("type") == "output_text":
                        text_parts.append(content_part.get("text", ""))
                    elif isinstance(content_part, str):
                        text_parts.append(content_part)
                content = "\n".join(text_parts)
            elif isinstance(content, str):
                pass
            else:
                content = str(content)

            if content:
                content_parts.append(content)

        elif element_type == "function_call":
            # Handle function calls
            try:
                name = element.get("name", "")
                call_id = element.get("call_id", "")
                arguments_str = element.get("arguments", "{}")

                # Parse arguments if it's a string
                if isinstance(arguments_str, str):
                    try:
                        arguments_obj = json.loads(arguments_str) if arguments_str else {}
                    except json.JSONDecodeError:
                        arguments_obj = {}
                        warnings.warn(
                            f"Failed to parse function arguments: {arguments_str}. Using empty dict.",
                            UserWarning,
                            stacklevel=2,
                        )
                else:
                    arguments_obj = arguments_str if isinstance(arguments_str, dict) else {}

                # Format function call as JSON within XML tags (matching original format)
                function_call = {
                    "name": name,
                    "arguments": arguments_obj,
                }

                formatted_call = f"{_TOOL_CALL_TAG_START}\n{json.dumps(function_call, indent=2)}\n{_TOOL_CALL_TAG_END}"
                content_parts.append(formatted_call)

            except (KeyError, TypeError) as e:
                warnings.warn(
                    f"Error formatting function call in response: {e}. Skipping this function call.",
                    UserWarning,
                    stacklevel=2,
                )

        elif element_type == "function_call_output":
            # Handle function call outputs/results
            try:
                call_id = element.get("call_id", "")
                name = element.get("name", "function")
                output = element.get("output", "")

                # Format function response as JSON within XML tags
                tool_response = {
                    "name": name,
                    "call_id": call_id,
                    "output": output,
                }

                formatted_response = (
                    f"{_TOOL_RESPONSE_TAG_START}\n{json.dumps(tool_response, indent=2)}\n{_TOOL_RESPONSE_TAG_END}"
                )
                content_parts.append(formatted_response)

            except (KeyError, TypeError) as e:
                warnings.warn(
                    f"Error formatting function call output in response: {e}. Skipping this output.",
                    UserWarning,
                    stacklevel=2,
                )
        elif element_type:
            warnings.warn(
                f"Unknown response element type: {element_type}. Skipping this element.",
                UserWarning,
                stacklevel=2,
            )
    return "\n".join(content_parts)


def form_response_string_responses_api_from_response(response: ModelResponse) -> str:
    """
    Form a single string representing the response from an OpenAI Responses API Response object.

    This function extracts the output list from a Response object and formats it using
    the list-based formatting function.

    Args:
        response (Response): A Response object returned by OpenAI's Responses API.
            The function uses the output list from the response.

    Returns:
        str: A formatted string containing the response content and any function calls.
             Function calls are formatted as XML tags containing JSON with function
             name and arguments.

    Raises:
        ImportError: If openai is not installed.
        AttributeError: If the response object doesn't have the expected structure.
    """
    # Convert response output to list of dictionaries
    output_list = []
    for output_item in response.output:
        if hasattr(output_item, "model_dump"):
            # Handle pydantic models
            output_list.append(output_item.model_dump())
        elif hasattr(output_item, "__dict__"):
            # Handle objects with __dict__
            output_list.append(output_item.__dict__)
        elif isinstance(output_item, dict):
            # Handle dictionaries
            output_list.append(output_item)
        else:
            # Convert to dict if possible
            try:
                output_list.append(dict(output_item))
            except (TypeError, ValueError):
                warnings.warn(
                    f"Could not convert output item to dict: {type(output_item)}. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )

    return form_response_string_responses_api_list(output_list)


def get_tool_result_as_text(messages: list[ChatCompletionMessageParam], tool_name: str) -> str:
    """
    Extract tool result as text for a specific tool name in the current chat turn.

    Searches through OpenAI ChatCompletion messages to find tool results matching
    the given tool name in the current turn only.

    Args:
        messages: List of OpenAI ChatCompletion conversation messages
        tool_name: Name of the tool to extract results for

    Returns:
        Concatenated text content from matching tool results in current turn
    """
    # 1. Find the last user message (start of current turn)
    last_user_idx = None
    for i in reversed(range(len(messages))):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return ""

    # 2. Find tool call IDs from assistant messages after the last user message
    tool_ids = set()
    for i in range(last_user_idx + 1, len(messages)):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_msg = cast(ChatCompletionAssistantMessageParam, msg)
            for tool_call in assistant_msg["tool_calls"]:
                # Only handle function tool calls, skip custom tool calls
                if tool_call.get("type") == "function":
                    # Type narrow to function tool call
                    func_tool_call = cast(Any, tool_call)  # Use Any to avoid union issues
                    if (
                        func_tool_call.get("function")
                        and func_tool_call["function"].get("name") == tool_name
                        and func_tool_call.get("id")
                    ):
                        tool_ids.add(func_tool_call["id"])

    # 3. Collect content from tool messages in current turn with matching tool_call_ids
    texts = []
    for i in range(last_user_idx + 1, len(messages)):
        msg = messages[i]
        if msg.get("role") == "tool" and msg.get("tool_call_id") in tool_ids:
            content = msg.get("content", "")
            if content:
                texts.append(str(content))

    return "\n\n".join(texts)
