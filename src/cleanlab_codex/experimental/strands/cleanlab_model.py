from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterable, Optional, Type, TypeVar, Union, cast

from cleanlab_tlm.utils.chat import form_response_string_chat_completions_api
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from strands.models.model import Model  # type: ignore[import-not-found]
from strands.models.openai import OpenAIModel  # type: ignore[import-not-found]
from strands.types.tools import ToolSpec, ToolUse  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from codex.types.project_validate_response import ProjectValidateResponse

    from cleanlab_codex import Project
    from strands.agent.agent import Agent  # type: ignore[import-not-found]
    from strands.types.content import ContentBlock, Messages  # type: ignore[import-not-found]
    from strands.types.streaming import StreamEvent  # type: ignore[import-not-found]

T = TypeVar("T")
OPENAI_TEXT_PART_TYPES = {"text", "output_text", "input_text"}


# ============ Helper Functions ============
def get_tool_result_as_text(messages: Messages, tool_name: str) -> str:
    """
    Extract tool result as text for a specific tool name in the current chat turn.

    Searches through messages to find tool results matching the given tool name
    and extracts all text, JSON, image, and document content as formatted strings.

    Args:
        messages: List of Strands conversation messages
        tool_name: Name of the tool to extract results for

    Returns:
        Concatenated text content from matching tool results
    """
    if not messages:
        return ""

    # 1. Find the last tool/user message (current turn)
    last_user_idx = get_latest_user_or_tool_message_index(messages)
    last_user_msg = messages[last_user_idx] if last_user_idx is not None else None

    if not last_user_msg or last_user_idx is None:
        return ""

    # 2. Find the immediately preceding assistant message (tool calls)
    prev_assistant_msg = messages[last_user_idx - 1] if last_user_idx > 0 else None
    tool_ids = set()
    if prev_assistant_msg and prev_assistant_msg.get("role") == "assistant":
        for block in prev_assistant_msg.get("content", []):
            if "toolUse" in block:
                tool_use = block["toolUse"]
                if tool_use.get("name") == tool_name:
                    tool_ids.add(tool_use.get("toolUseId"))

    # 3. Collect ALL content from the last user message for matching tool IDs
    texts = []
    for block in last_user_msg.get("content", []):
        if "toolResult" in block:
            result = block["toolResult"]
            if result.get("toolUseId") in tool_ids:
                for c in result.get("content", []):
                    if "text" in c:
                        texts.append(c["text"])
                    elif "json" in c:
                        import json

                        texts.append(json.dumps(c["json"], indent=2))
                    elif "image" in c:
                        texts.append(f"[Image content: {c['image'].get('format', 'unknown format')}]")
                    elif "document" in c:
                        texts.append(f"[Document content: {c['document'].get('name', 'unknown document')}]")

    return "\n\n".join(texts)


# ============ Strands to OpenAI Conversion Utils ============


def convert_strands_tools_to_openai_format(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert Strands tool specifications to OpenAI function calling format.

    Transforms tool specs with inputSchema into OpenAI's expected format
    with function name, description, and parameters schema.

    Args:
        tool_specs: List of Strands tool specifications

    Returns:
        List of tools formatted for OpenAI function calling API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "parameters": tool_spec["inputSchema"]["json"],
            },
        }
        for tool_spec in tool_specs or []
    ]


def convert_strands_messages_for_cleanlab(messages: Messages) -> list[dict[str, Any]]:
    """
    Convert Strands message format to cleanlab validation format.

    Transforms complex Strands messages with content blocks into a simplified
    format where user content is flattened to strings and tool results are
    converted to separate tool messages.

    Args:
        messages: Messages in Strands format

    Returns:
        List of messages formatted for cleanlab validation
    """
    cleanlab_messages = []

    for message in messages:
        if message["role"] == "user":
            text_parts = []
            tool_results = []

            for content in message["content"]:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "toolResult" in content:
                    tool_results.append(content["toolResult"])

            if text_parts:
                text_content = "\n".join(text_parts)
                cleanlab_messages.append({"role": "user", "content": text_content})

            for tool_result in tool_results:
                content_parts = []
                for content_item in tool_result.get("content", []):
                    if "text" in content_item:
                        content_parts.append(content_item["text"])
                    elif "json" in content_item:
                        import json

                        content_parts.append(json.dumps(content_item["json"]))

                tool_content = "\n".join(content_parts) if content_parts else str(tool_result.get("content", []))

                cleanlab_messages.append(
                    {"role": "tool", "tool_call_id": tool_result.get("toolUseId", ""), "content": tool_content}
                )
        else:
            formatted_message = OpenAIModel.format_request_messages([message])[0]
            cleanlab_messages.append(formatted_message)

    return cleanlab_messages


def _convert_strands_content_to_openai_format(collected_content: list[ContentBlock]) -> dict[str, Any]:
    """Convert Strands content blocks to OpenAI assistant message format."""
    content_blocks = []
    tool_calls = []

    for block in collected_content:
        if "text" in block:
            content_blocks.append(OpenAIModel.format_request_message_content(block))
        if "toolUse" in block:
            tool_calls.append(OpenAIModel.format_request_message_tool_call(block["toolUse"]))

    assistant_message = {"role": "assistant"}
    if content_blocks:
        assistant_message["content"] = content_blocks  # type: ignore[assignment]
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls  # type: ignore[assignment]

    return assistant_message


def get_latest_user_message_content(messages: Messages) -> str:
    """
    Extract text content from the most recent user message.

    Searches backwards through messages to find the latest user message
    and returns the text content from its first content block.

    Args:
        messages: List of conversation messages

    Returns:
        Text content from latest user message, or empty string if none found
    """
    user_message = ""
    for msg in reversed(messages):
        if (
            msg.get("role") == "user"
            and msg.get("content")
            and len(msg["content"]) > 0
            and msg["content"][0].get("text")
        ):
            user_message = msg["content"][0]["text"].lower()

    return user_message


def get_latest_user_or_tool_message_index(messages: Messages) -> int | None:
    """
    Find the index of the most recent user message in the conversation.

    Searches backwards through messages to locate the latest message
    with role 'user'.

    Args:
        messages: List of conversation messages

    Returns:
        Index of the latest user message, or None if no user message found
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" or msg.get("role") == "tool":
            return i
    return None


def _extract_text(message: dict[str, Any]) -> str:
    """Return plain text from either the new content-parts format or the old string format."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in OPENAI_TEXT_PART_TYPES
        )
    return ""


# ============ Cleanlab Model Wrapper ============
class CleanlabModel(Model):  # type: ignore[misc]
    """Model wrapper that validates responses using Cleanlab."""

    def __init__(
        self,
        *,
        underlying_model: Model,
        cleanlab_project: Project,
        fallback_response: str = "Sorry I am unsure. You can try rephrasing your request.",
        context_retrieval_tools: list[str] | None = None,
        skip_validating_tool_calls: bool = True,
    ) -> None:
        """Initialize with an underlying model to wrap."""
        self.underlying_model = underlying_model
        self.cleanlab_project = cleanlab_project
        self.fallback_response = fallback_response
        self.context_retrieval_tools = context_retrieval_tools if context_retrieval_tools is not None else []
        self.skip_validating_tool_calls = (
            skip_validating_tool_calls  # Skip validation if there are tool calls in the response
        )
        self._agent_ref = None  # Set by agent after initialization

    def __getattr__(self, name: str) -> Any:
        """Delegate missing attributes to underlying model."""
        return getattr(self.underlying_model, name)

    def update_config(self, **model_config: Any) -> None:
        """
        Update configuration parameters for the underlying model.

        Passes through configuration changes to the wrapped model instance.

        Args:
            **model_config: Configuration parameters to update
        """
        self.underlying_model.update_config(**model_config)

    def get_config(self) -> Any:
        """
        Retrieve current configuration from the underlying model.

        Returns the configuration object from the wrapped model instance.

        Returns:
            Current model configuration
        """
        return self.underlying_model.get_config()

    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """
        Generate structured output using the underlying model.

        Passes through structured output requests to the wrapped model,
        yielding events as they are generated.

        Args:
            output_model: Pydantic model class for structured output
            prompt: Input messages for the model
            system_prompt: Optional system prompt
            **kwargs: Additional arguments for the model

        Yields:
            Dictionary events containing structured output results
        """
        async for event in self.underlying_model.structured_output(output_model, prompt, system_prompt, **kwargs):
            yield event

    def _get_session_id(self) -> Optional[str]:
        """Get session ID from agent if available."""
        if self._agent_ref and hasattr(self._agent_ref, "_session_manager"):
            session_manager = getattr(self._agent_ref, "_session_manager", None)
            if session_manager and hasattr(session_manager, "session_id"):
                return session_manager.session_id
        return None

    async def _collect_all_events(self, stream: AsyncIterable[StreamEvent]) -> list[StreamEvent]:
        """Collect all events from a stream."""
        return [event async for event in stream]

    def _reconstruct_message_from_events(self, collected_events: list[StreamEvent]) -> list[ContentBlock]:
        """Reconstruct complete message content from streaming events."""
        message_content = []
        current_text = ""
        current_tool_use = {}

        for event in collected_events:
            if "contentBlockStart" in event:
                start_data = event["contentBlockStart"]["start"]
                if "toolUse" in start_data:
                    tool_data = start_data["toolUse"]
                    current_tool_use = {"toolUseId": tool_data["toolUseId"], "name": tool_data["name"], "input": ""}
                elif "text" in start_data:
                    current_text = ""

            elif "contentBlockDelta" in event and "delta" in event["contentBlockDelta"]:
                delta = event["contentBlockDelta"]["delta"]

                if "text" in delta:
                    current_text += delta["text"]
                elif "toolUse" in delta:
                    current_tool_use["input"] += delta["toolUse"]["input"]

            elif "contentBlockStop" in event:
                if current_tool_use and current_tool_use.get("name"):
                    # Parse tool input JSON and create ToolUse object
                    try:
                        parsed_input = json.loads(current_tool_use["input"])
                    except (json.JSONDecodeError, ValueError):
                        parsed_input = {}

                    tool_use = ToolUse(
                        toolUseId=current_tool_use["toolUseId"],
                        name=current_tool_use["name"],
                        input=parsed_input,
                    )
                    message_content.append({"toolUse": tool_use})
                    current_tool_use = {}

                elif current_text:
                    message_content.append({"text": current_text})
                    current_text = ""

        return message_content

    async def _stream_validated_response(
        self,
        *,
        replacement_content: list[ContentBlock],
        final_stop_reason: str,
        final_usage: Optional[dict[str, Any]],
        final_metrics: Optional[dict[str, Any]],
        validation_results: ProjectValidateResponse,
        is_replaced: bool,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream the validated response back to caller."""
        # Stream the validated response
        yield {"messageStart": {"role": "assistant"}}

        # Stream each content block
        for block in replacement_content:
            if "text" in block:
                # Stream text block
                yield {"contentBlockStart": {"start": {"text": ""}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
                yield {"contentBlockStop": {}}

            elif "toolUse" in block:
                # Stream tool use block
                tool = block["toolUse"]
                yield {
                    "contentBlockStart": {"start": {"toolUse": {"toolUseId": tool["toolUseId"], "name": tool["name"]}}}
                }
                # Stream tool input as JSON string
                tool_input_json = json.dumps(tool["input"])
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": tool_input_json}}}}
                yield {"contentBlockStop": {}}

        # End the message
        yield {"messageStop": {"stopReason": final_stop_reason or "end_turn"}}

        # Include metadata
        yield {
            "metadata": {
                "usage": final_usage
                or {
                    "inputTokens": -1,
                    "outputTokens": -1,
                    "totalTokens": -1,
                },
                "metrics": final_metrics or {"latencyMs": -1},
                "cleanlab": {
                    "validation_results": validation_results.model_dump(),
                    "is_replaced": is_replaced,
                    "stop_reason": final_stop_reason,
                },
            }
        }

    def _cleanlab_cleanup_messages(self) -> None:
        """Remove tool calls and results from conversation when response was replaced."""
        if not self._agent_ref:
            return

        # Access messages directly from the agent (not from conversation_manager)
        messages = self._agent_ref.messages
        if not messages:
            return

        messages_to_remove = []

        # Find the last assistant message with tool calls
        last_assistant_with_tools_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if (
                msg.get("role") == "assistant"
                and "content" in msg
                and any("toolUse" in block for block in msg["content"])
            ):
                last_assistant_with_tools_idx = i
                break

        if last_assistant_with_tools_idx is not None:
            # Mark the assistant message with tool calls for removal
            messages_to_remove.append(last_assistant_with_tools_idx)

            # Find subsequent user messages that contain only tool results
            for i in range(last_assistant_with_tools_idx + 1, len(messages)):
                msg = messages[i]

                if msg.get("role") == "user" and "content" in msg:
                    content_blocks = msg["content"]
                    has_tool_results = all("toolResult" in block for block in content_blocks if block)

                    if has_tool_results:
                        messages_to_remove.append(i)
                    else:
                        # Stop at the first non-tool-result message
                        break
                else:
                    break

        # Remove messages in reverse order to maintain indices
        for idx in sorted(messages_to_remove, reverse=True):
            if 0 <= idx < len(messages):
                messages.pop(idx)

    def _has_recent_tool_calls(self) -> bool:
        """Check if there are recent tool calls that should be cleaned up."""
        if not self._agent_ref or not self._agent_ref.messages:
            return False

        messages = self._agent_ref.messages
        # Look for assistant messages with tool calls, starting from the end
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            # If we hit a user message that's not a tool result, we've gone too far back
            if msg.get("role") == "user" and "content" in msg:
                content = msg["content"]
                # Check if all blocks in this user message are tool results
                has_only_tool_results = all("toolResult" in block for block in content if block)
                if not has_only_tool_results:
                    break

            # Check if this is an assistant message with tool calls
            elif msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                has_tool_use = any("toolUse" in block for block in content)
                if has_tool_use:
                    return True
        return False

    def _cleanlab_validate(
        self,
        messages: Messages,
        collected_content: list[ContentBlock],
        tool_specs: list[dict[str, Any]],
        stop_reason: str,
    ) -> tuple[ProjectValidateResponse, list[ContentBlock], bool]:
        """Validate response and return (results, content, is_replaced)."""
        session_id = self._get_session_id()

        validate_fields = self.cleanlab_get_validate_fields(messages)

        openai_collected_content = _convert_strands_content_to_openai_format(collected_content)

        if "content" in openai_collected_content:  # TODO: Remove after update to handle new OpenAI response format
            openai_collected_content["content"] = _extract_text(openai_collected_content)
        if (
            len(openai_collected_content.get("tool_calls", [])) > 0 and self.skip_validating_tool_calls
        ):  # assistant message with tool calls
            eval_scores = {"trustworthiness": 1.0, "response_helpfulness": 1.0}
            validation_results = self.cleanlab_project.validate(
                response=form_response_string_chat_completions_api(openai_collected_content),
                messages=cast(list[ChatCompletionMessageParam], convert_strands_messages_for_cleanlab(messages)),
                tools=cast(list[ChatCompletionToolParam], convert_strands_tools_to_openai_format(tool_specs))
                if tool_specs
                else None,
                metadata={"thread_id": session_id, "stop_reason": stop_reason},
                eval_scores=eval_scores,
                **validate_fields,
            )
        else:
            validation_results = self.cleanlab_project.validate(
                response=form_response_string_chat_completions_api(openai_collected_content),
                messages=cast(list[ChatCompletionMessageParam], convert_strands_messages_for_cleanlab(messages)),
                tools=cast(list[ChatCompletionToolParam], convert_strands_tools_to_openai_format(tool_specs))
                if tool_specs
                else None,
                metadata={"thread_id": session_id, "stop_reason": stop_reason},
                **validate_fields,
            )

        final_response, is_replaced = self.cleanlab_get_final_response(
            validation_results,
            initial_response=collected_content,
            fallback_response=self.fallback_response,
        )

        if self._agent_ref:
            self._agent_ref.state.set("cleanlab_validation_results", validation_results.model_dump())
            self._agent_ref.state.set("initial_model_response", collected_content)

        return validation_results, final_response, is_replaced

    def _get_context_as_string(self, messages: Messages) -> str:
        """Extract context from tool results in the agent's messages."""
        context_parts = ""
        for tool_name in self.context_retrieval_tools:
            tool_result_text = get_tool_result_as_text(messages, tool_name)
            if tool_result_text:
                context_parts += f"Context from tool {tool_name}:\n{tool_result_text}\n\n"

        return context_parts

    def set_agent_reference(self, agent: Agent) -> None:
        """
        Set reference to the agent instance for accessing session information.

        The agent reference is used to access session IDs, cleanlab validation results,
        and perform message cleanup operations when responses are replaced.

        Args:
            agent: The agent instance using this model
        """
        self._agent_ref = agent

    @staticmethod
    def cleanlab_get_final_response(
        results: ProjectValidateResponse, initial_response: Any, fallback_response: str
    ) -> tuple[list[ContentBlock], bool]:
        """
        Determine the final response content based on cleanlab validation results.

        Checks validation results for expert answers or guardrail triggers,
        returning either the original response or a replacement.

        Args:
            results: Validation results from cleanlab
            initial_response: Original model response content
            fallback_response: Fallback text for guardrailed responses

        Returns:
            Tuple of (final_content, was_replaced_flag)
        """

        is_replaced = False
        final_response = initial_response
        if results.expert_answer and results.escalated_to_sme:
            final_response = results.expert_answer
            is_replaced = True
        elif results.should_guardrail:
            final_response = fallback_response
            is_replaced = True

        if is_replaced:
            return [{"text": final_response}], is_replaced
        return initial_response, is_replaced

    def cleanlab_get_validate_fields(self, messages: Messages) -> dict[str, Any]:
        """
        Extract query and context fields from Strands messages for cleanlab validation.

        Processes conversation messages to extract the user query and any
        contextual information from specified tool results.

        Args:
            messages: Conversation messages to process

        Returns:
            Dictionary with 'query' and 'context' fields for validation
        """
        user_message = get_latest_user_message_content(messages)

        context = self._get_context_as_string(messages)
        return {
            "query": user_message,
            "context": context,
        }

    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """
        Stream model responses with cleanlab validation and potential replacement.

        Collects the complete response from the underlying model, validates it
        through cleanlab, and streams either the original or replacement content.
        Handles tool call cleanup when responses are replaced.

        Args:
            messages: Input conversation messages
            tool_specs: Available tool specifications
            system_prompt: Optional system prompt
            **kwargs: Additional model arguments

        Yields:
            Stream events containing validated response content
        """
        # Warn if agent reference is not set - needed for tool call cleanup functionality
        if not self._agent_ref:
            warnings.warn(
                "CleanlabModel missing agent reference. Call set_agent_reference(agent) for full functionality.",
                UserWarning,
                stacklevel=2,
            )

        # Step 1: Collect the complete response from underlying model
        message_content = []
        final_stop_reason = "end_turn"
        final_usage = None
        final_metrics = None

        try:
            # Wait for complete streaming response - collect all events
            stream = self.underlying_model.stream(messages, tool_specs, system_prompt, **kwargs)
            collected_events = await self._collect_all_events(stream)

            # Extract metadata from collected events
            for event in collected_events:
                if "messageStop" in event:
                    final_stop_reason = event["messageStop"]["stopReason"]
                elif "metadata" in event:
                    final_usage = event["metadata"].get("usage")
                    final_metrics = event["metadata"].get("metrics")

            # Reconstruct the complete message content from collected events
            message_content = self._reconstruct_message_from_events(collected_events)

        except (TypeError, ValueError, RuntimeError) as e:
            warnings.warn(
                f"Error collecting model response: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Step 2: Validate the complete message content once
        validation_results, replacement_content, is_replaced = self._cleanlab_validate(
            messages, message_content, tool_specs or [], final_stop_reason
        )

        # Step 3: Clean up messages history of current chat turn's tool calls if response was replaced with fallback or expert answer
        # Check if there are recent tool calls in the conversation (not just current stream)
        has_recent_tool_calls = self._has_recent_tool_calls()
        if is_replaced and has_recent_tool_calls:
            self._cleanlab_cleanup_messages()

        # Step 4: Stream the validated response
        async for event in self._stream_validated_response(
            replacement_content=replacement_content,
            final_stop_reason=final_stop_reason,
            final_usage=final_usage,
            final_metrics=final_metrics,
            validation_results=validation_results,
            is_replaced=is_replaced,
        ):
            yield event
