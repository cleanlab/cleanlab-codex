from __future__ import annotations

from typing import Any, ClassVar, Optional

from cleanlab_codex.project import Project


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
        project: Project,
        *,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ):
        self._project = project
        self._fallback_answer = fallback_answer

    @classmethod
    def from_access_key(
        cls,
        access_key: str,
        *,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from an access key. The project ID that the CodexTool will use is the one that is associated with the access key."""
        project = Project.from_access_key(access_key)
        return cls(
            project=project,
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
        """Asks an all-knowing advisor this question in cases where it cannot be answered from the provided Context.

        Args:
            question: The question to ask the advisor. This should be the same as the original user question, except in cases where the user question is missing information that could be additionally clarified.

        Returns:
            The answer to the question. If the answer is not available, this returns a fallback answer or None.
        """
        return self._project.query(question, fallback_answer=self._fallback_answer)[0]

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts the tool to an OpenAI tool."""
        from cleanlab_codex.utils import format_as_openai_tool

        return format_as_openai_tool(
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            tool_properties=self._tool_properties,
            required_properties=self._tool_requirements,
        )

    def to_smolagents_tool(self) -> Any:
        """Converts the tool to a smolagents tool."""
        from cleanlab_codex.utils.smolagents import CodexTool as SmolagentsCodexTool

        return SmolagentsCodexTool(
            query=self.query,
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            inputs=self._tool_properties,
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

    def to_langchain_tool(self) -> Any:
        """Converts the tool to a [LangChain tool](https://python.langchain.com/docs/concepts/tools/).

        Note: You must have the [`langchain` library installed](https://python.langchain.com/docs/concepts/architecture/) to use this method.
        """
        from langchain_core.tools.structured import StructuredTool

        from cleanlab_codex.utils.langchain import create_args_schema

        return StructuredTool.from_function(
            func=self.query,
            name=self._tool_name,
            description=self._tool_description,
            args_schema=create_args_schema(
                name=self._tool_name,
                func=self.query,
                tool_properties=self._tool_properties,
            ),
        )

    def to_aws_converse_tool(self) -> Any:
        """Converts the tool to an [AWS Converse API tool](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-inference-call.html).

        Note: You must have the [`boto3` library installed](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html) (AWS SDK for Python) to use this method.
        """
        from cleanlab_codex.utils.aws import format_as_aws_converse_tool

        return format_as_aws_converse_tool(
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            tool_properties=self._tool_properties,
            required_properties=self._tool_requirements,
        )
