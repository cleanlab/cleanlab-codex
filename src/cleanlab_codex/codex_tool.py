"""Tool abstraction for Cleanlab Codex."""

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
        project_id: Optional[str] = None,
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
        project_id: Optional[str] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from an access key. The project ID that the CodexTool will use is the one that is associated with the access key.

        Args:
            access_key (str): The access key for the Codex project.
            project_id (str, optional): The ID of the project to use. If not provided, the project ID will be inferred from the access key. If provided, the project ID must be the ID of the project that the access key is associated with.
            fallback_answer (str, optional): The fallback answer to use if the Codex project cannot answer the question.

        Returns:
            CodexTool: The CodexTool.
        """
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
        project_id: Optional[str] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from a Codex client.
        If the Codex client is initialized with a project access key, the CodexTool will use the project ID that is associated with the access key.
        If the Codex client is initialized with a user API key, a project ID must be provided.

        Args:
            codex_client (Codex): The Codex client to use.
            project_id (str, optional): The ID of the project to use. If not provided and the Codex client is authenticated with a project-level access key, the project ID will be inferred from the access key.
            fallback_answer (str, optional): The fallback answer to use if the Codex project cannot answer the question.

        Returns:
            CodexTool: The CodexTool.
        """
        return cls(
            codex_client=codex_client,
            project_id=project_id,
            fallback_answer=fallback_answer,
        )

    @property
    def tool_name(self) -> str:
        """The name to use for the tool when passing to an LLM. This is the name the LLM will use when determining whether to call the tool.

        Note: We recommend using the default tool name which we've benchmarked. Only override this if you have a specific reason."""
        return self._tool_name

    @tool_name.setter
    def tool_name(self, value: str) -> None:
        """Sets the name to use for the tool when passing to an LLM."""
        self._tool_name = value

    @property
    def tool_description(self) -> str:
        """The description to use for the tool when passing to an LLM. This is the description that the LLM will see when determining whether to call the tool.

        Note: We recommend using the default tool description which we've benchmarked. Only override this if you have a specific reason."""
        return self._tool_description

    @tool_description.setter
    def tool_description(self, value: str) -> None:
        """Sets the description to use for the tool when passing to an LLM."""
        self._tool_description = value

    @property
    def fallback_answer(self) -> Optional[str]:
        """The fallback answer to use if the Codex project cannot answer the question. This will be returned from by the tool if the Codex project does not have an answer to the question."""
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
            The answer to the question if available. If no answer is available, the fallback answer is returned if provided, otherwise None is returned.
        """
        return self._codex_client.query(question, project_id=self._project_id, fallback_answer=self._fallback_answer)[0]

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts the tool to an [OpenAI tool](https://platform.openai.com/docs/guides/function-calling#defining-functions)."""
        from cleanlab_codex.utils import format_as_openai_tool

        return format_as_openai_tool(
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            tool_properties=self._tool_properties,
            required_properties=self._tool_requirements,
        )

    def to_smolagents_tool(self) -> Any:
        """Converts the tool to a [smolagents tool](https://huggingface.co/docs/smolagents/reference/tools#smolagents.Tool).

        Note: You must have the [`smolagents` library installed](https://github.com/huggingface/smolagents/tree/main?tab=readme-ov-file#quick-demo) to use this method.
        """
        from cleanlab_codex.utils.smolagents import CodexTool as SmolagentsCodexTool

        return SmolagentsCodexTool(
            query=self.query,
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            inputs=self._tool_properties,
        )

    def to_llamaindex_tool(self) -> Any:
        """Converts the tool to a [LlamaIndex FunctionTool](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/#functiontool).

        Note: You must have the [`llama-index` library installed](https://docs.llamaindex.ai/en/stable/getting_started/installation/) to use this method.
        """
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
