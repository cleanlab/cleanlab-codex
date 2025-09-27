"""Cleanlab response rewriting hook V3 - Fixed text extraction for all response types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agents.items import ModelResponse, TResponseInputItem
    from codex.types.project_validate_response import ProjectValidateResponse
    from openai.types.chat import ChatCompletionMessageParam

    from cleanlab_codex import Project

import secrets

from agents import FunctionTool
from agents.lifecycle import RunHooks
from agents.models.chatcmpl_converter import Converter
from agents.run_context import RunContextWrapper, TContext
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from cleanlab_codex.experimental.openai_agents.utils import (
    form_response_string_responses_api_from_response,
    get_tool_result_as_text,
)


def _cleanlab_string_to_response_output_message(text: str, message_id: str | None = None) -> ResponseOutputMessage:
    if message_id is None:
        message_id = f"msg_cleanlab{secrets.token_hex(16)}"  # TODO: Add support for marking cleanlab responses beyond adding cleanlab to ID
    return ResponseOutputMessage(
        id=message_id,
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
        role="assistant",
        type="message",
        status="completed",
    )


def _rewrite_response_content_inplace(response: ModelResponse, new_content: str) -> None:
    """Rewrite the response content and remove tool calls."""
    response.output.clear()
    new_message_raw = _cleanlab_string_to_response_output_message(new_content)
    response.output.append(new_message_raw)


class CleanlabHook(RunHooks[TContext]):
    """V3 hook with comprehensive text extraction for all OpenAI response types."""

    def __init__(
        self,
        *,
        cleanlab_project: Project,
        fallback_response: str = "Sorry I am unsure. You can try rephrasing your request.",
        skip_validating_tool_calls: bool = False,
        context_retrieval_tools: list[str] | None = None,
        validate_every_response: bool = True,
    ) -> None:
        """Initialize Cleanlab response rewriter hook V3."""
        super().__init__()
        self.cleanlab_project = cleanlab_project
        self.fallback_response = fallback_response
        self.skip_validating_tool_calls = skip_validating_tool_calls
        self.context_retrieval_tools = context_retrieval_tools or []
        self.validate_every_response = validate_every_response

        # Populated by on_llm_start with actual conversation history
        self._conversation_history: list[ChatCompletionMessageParam] = []
        self._system_prompt: Optional[str] = None
        self._latest_response_text: Optional[str] = None

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Any,  # noqa: ARG002
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """Capture the conversation history being sent to the LLM and set up context for storing results."""
        raw_messages = Converter.items_to_messages(input_items)
        self._conversation_history = raw_messages
        self._system_prompt = system_prompt
        if context.context is None:
            context.context = type("CleanlabContext", (), {})()

    async def on_llm_end(self, context: RunContextWrapper[TContext], agent: Any, response: ModelResponse) -> None:
        """Intercept and potentially rewrite model response before tool execution."""
        # Perform Cleanlab validation with actual conversation history
        validation_result = await self._cleanlab_validate(response, context, agent)

        # Rewrite response if validation indicates we should
        await self.cleanlab_get_final_response(response, validation_result)

        # Store validation result in context
        context.context.latest_cleanlab_validation_result = validation_result  # type: ignore[attr-defined]
        context.context.latest_initial_response_text = self._get_latest_response_text(response)  # type: ignore[attr-defined]

        # Clear state vars
        self._latest_response_text = None

    def _should_validate_response(self, response: ModelResponse) -> bool:
        """Determine if this response should be validated with Cleanlab."""
        if self.skip_validating_tool_calls and self._response_has_tool_calls(response):
            return False
        return self._response_has_content(response)

    def _response_has_tool_calls(self, response: ModelResponse) -> bool:
        """Check if model response contains tool calls."""
        for item in response.output:
            # Check for tool calls in various formats
            if hasattr(item, "tool_calls") and item.tool_calls:
                return True
            if hasattr(item, "type") and "function_call" in str(item.type).lower():
                return True
            if "FunctionToolCall" in type(item).__name__:
                return True
        return False

    def _response_has_content(self, response: ModelResponse) -> bool:
        """Check if response has content that can be validated."""
        return bool(self._get_latest_response_text(response).strip())

    def _get_latest_response_text(self, response: ModelResponse) -> str:
        if self._latest_response_text is None:
            self._latest_response_text = form_response_string_responses_api_from_response(response)
        return self._latest_response_text

    def _get_latest_user_query(self) -> str:
        """Extract the most recent user query from the actual conversation history."""
        for item in reversed(self._conversation_history):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _get_context_as_string(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Extract context from tool results in the agent's messages."""
        context_parts = ""
        for tool_name in self.context_retrieval_tools:
            tool_result_text = get_tool_result_as_text(messages, tool_name)
            if tool_result_text:
                context_parts += f"Context from tool {tool_name}:\n{tool_result_text}\n\n"

        return context_parts

    async def _cleanlab_validate(
        self, response: ModelResponse, context: RunContextWrapper[TContext], agent: Any
    ) -> ProjectValidateResponse | None:
        """Validate the model response using Cleanlab with actual conversation history."""
        # Step 1 - Convert hook items to Cleanlab format
        tools_dict = (
            [Converter.tool_to_openai(tool) for tool in agent.tools if isinstance(tool, FunctionTool)]
            if agent.tools
            else None
        )
        cleanlab_messages = list(self._conversation_history)
        if self._system_prompt:
            cleanlab_messages.insert(
                0,
                {
                    "content": self._system_prompt,
                    "role": "system",
                },
            )

        session_id = getattr(context, "session_id", None) or "unknown"

        # Step 2 - Get additional validation fields
        validate_fields = self.cleanlab_get_validate_fields(cleanlab_messages)
        eval_scores = None
        if not self._should_validate_response(response):
            eval_scores = {
                "trustworthiness": 1.0,
                "response_helpfulness": 1.0,
                "context_sufficiency": 1.0,
                "query_ease": 1.0,
                "response_groundedness": 1.0,
            }

        # Step 3 - Run validation
        return self.cleanlab_project.validate(
            response=self._get_latest_response_text(response),
            messages=cleanlab_messages,
            tools=tools_dict,
            metadata={
                "thread_id": session_id,
                "agent_name": getattr(agent, "name", "unknown"),
            },
            eval_scores=eval_scores,
            **validate_fields,
        )

    def cleanlab_get_validate_fields(self, messages: list[ChatCompletionMessageParam]) -> dict[str, Any]:
        """
        Extract query and context fields from Strands messages for cleanlab validation.

        Processes conversation messages to extract the user query and any
        contextual information from specified tool results.

        Args:
            messages: Conversation messages to process

        Returns:
            Dictionary with 'query' and 'context' fields for validation
        """
        user_message = self._get_latest_user_query()
        context = self._get_context_as_string(messages)
        return {
            "query": user_message,
            "context": context,
        }

    async def cleanlab_get_final_response(
        self, response: ModelResponse, validation_result: ProjectValidateResponse | None
    ) -> None:
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
        if validation_result is None:
            return

        replacement_text = None
        if validation_result.expert_answer:
            replacement_text = validation_result.expert_answer
        elif validation_result.should_guardrail:
            replacement_text = self.fallback_response

        if replacement_text:
            _rewrite_response_content_inplace(response, replacement_text)
