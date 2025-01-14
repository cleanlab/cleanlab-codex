from __future__ import annotations

from typing import Any, ClassVar, Optional

from cleanlab_codex.codex import Codex


class CodexTool:
    """A tool that connects to a Codex project to answer questions."""

    _tool_name = "ask_advisor"
    _tool_description = "Asks an all-knowing advisor this query in cases where it cannot be answered from the provided Context. If the answer is avalible, this returns None."
    _tool_properties: ClassVar[dict[str, Any]] = {
        "question": {
            "type": "string",
            "description": "The question to ask the advisor. This should be the same as the original user question, except in cases where the user question is missing information that could be additionally clarified.",
        }
    }
    _tool_requirements: ClassVar[list[str]] = ["question"]
    DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."

    def __init__(
        self,
        codex_client: Codex,
        *,
        project_id: Optional[int] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ):
        self._codex_client = codex_client
        self._project_id = project_id
        self._fallback_answer = fallback_answer

    @classmethod
    def from_access_key(
        cls,
        access_key: str,
        *,
        project_id: Optional[int] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from an access key. The project ID that the CodexTool will use is the one that is associated with the access key."""
        return cls(
            codex_client=Codex(key=access_key),
            project_id=project_id,
            fallback_answer=fallback_answer,
        )

    @classmethod
    def from_client(
        cls,
        codex_client: Codex,
        *,
        project_id: Optional[int] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from a Codex client.
        If the Codex client is initialized with a project access key, the CodexTool will use the project ID that is associated with the access key.
        If the Codex client is initialized with a user API key, a project ID must be provided.
        """
        return cls(
            codex_client=codex_client,
            project_id=project_id,
            fallback_answer=fallback_answer,
        )

    @property
    def tool_name(self) -> str:
        """The name to use for the tool when passing to an LLM."""
        return self._tool_name

    @property
    def tool_description(self) -> str:
        """The description to use for the tool when passing to an LLM."""
        return self._tool_description

    @property
    def fallback_answer(self) -> Optional[str]:
        """The fallback answer to use if the Codex project cannot answer the question."""
        return self._fallback_answer

    @fallback_answer.setter
    def fallback_answer(self, value: Optional[str]) -> None:
        """Sets the fallback answer to use if the Codex project cannot answer the question."""
        self._fallback_answer = value

    def query(self, question: str) -> Optional[str]:
        """Asks an all-knowing advisor this question in cases where it cannot be answered from the provided Context. If the answer is not available, this returns a fallback answer or None.

        Args:
            question: The question to ask the advisor. This should be the same as the original user question, except in cases where the user question is missing information that could be additionally clarified.

        Returns:
            The answer to the question, or None if the answer is not available.
        """
        return self._codex_client.query(question, project_id=self._project_id, fallback_answer=self._fallback_answer)[0]

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts the tool to an OpenAI tool."""
        from cleanlab_codex.utils import format_as_openai_tool

        return format_as_openai_tool(
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            tool_properties=self._tool_properties,
            required_properties=self._tool_requirements,
        )

    def to_llamaindex_tool(self) -> Any:
        """Converts the tool to a LlamaIndex FunctionTool."""
        from llama_index.core.tools import FunctionTool

        from cleanlab_codex.utils.llamaindex import get_function_schema

        return FunctionTool.from_defaults(
            fn=self.query,
            name=self._tool_name,
            description=self._tool_description,
            fn_schema=get_function_schema(
                name=self._tool_name,
                func=self.query,
                tool_properties=self._tool_properties,
            ),
        )
