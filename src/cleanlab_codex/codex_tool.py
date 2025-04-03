"""Tool abstraction for Cleanlab Codex."""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import Annotated

from cleanlab_codex.internal.analytics import IntegrationType, _AnalyticsMetadata
from cleanlab_codex.project import Project
from cleanlab_codex.utils.errors import MissingDependencyError
from cleanlab_codex.utils.function import (
    pydantic_model_from_function,
    required_properties_from_model,
)


class CodexTool:
    """A tool that connects to a Codex project to answer questions."""

    _tool_name = "consult_codex"
    _tool_description = "Consults a database that contains answers to any possible question. If the answer is not available, this returns None."
    DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."

    def __init__(
        self,
        project: Project,
        *,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ):
        """Initialize a CodexTool.

        Args:
            project (Project): The Codex project to use for this tool.
            fallback_answer (str, optional): The fallback answer to use if the Codex project cannot answer the question. A default will be used if not provided.
        """
        self._project = project
        self._fallback_answer = fallback_answer
        self._tool_function_schema = pydantic_model_from_function(self._tool_name, self.query)
        self._tool_properties = self._tool_function_schema.model_json_schema()["properties"]
        self._tool_requirements = required_properties_from_model(self._tool_function_schema)

    @classmethod
    def from_access_key(
        cls,
        access_key: str,
        *,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
    ) -> CodexTool:
        """Creates a CodexTool from an access key. The CodexTool will use the project associated with the access key provided.

        Args:
            access_key (str): The access key for the Codex project.
            fallback_answer (str, optional): The fallback answer to use if the Codex project cannot answer the question.

        Returns:
            CodexTool: The CodexTool.
        """
        project = Project.from_access_key(access_key)
        return cls(
            project=project,
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
        """The fallback answer to use if the Codex project cannot answer the question. This will be returned by the tool if the Codex project does not have an answer to the question."""
        return self._fallback_answer

    @fallback_answer.setter
    def fallback_answer(self, value: Optional[str]) -> None:
        """Sets the fallback answer to use if the Codex project cannot answer the question."""
        self._fallback_answer = value

    def query(
        self,
        question: Annotated[
            str,
            "The query to search in the database. It should match the original user query unless clarification is needed (for instance to account for prior user messages), in which case changes to the query should be minimal.",
        ],
    ) -> Optional[str]:
        """Consults a database that contains answers to any possible question. If the answer is not available, this returns None.

        Args:
            question (str): The query to search in the database. It should match the original user query unless clarification is needed (for instance to account for prior user messages), in which case changes to the query should be minimal.

        Returns:
            The answer to the question if available. If no answer is available, this returns a fallback answer or None.
        """
        return self._project.query(
            question=question,
            fallback_answer=self._fallback_answer,
            _analytics_metadata=_AnalyticsMetadata(integration_type=IntegrationType.TOOL),
        )[0]

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts the tool to the expected format for an [OpenAI function tool](https://platform.openai.com/docs/guides/function-calling).
        See more information on defining functions for OpenAI tool calls [here](https://platform.openai.com/docs/guides/function-calling#defining-functions)."""
        from cleanlab_codex.utils import format_as_openai_tool

        return format_as_openai_tool(
            tool_name=self._tool_name,
            tool_description=self._tool_description,
            tool_properties=self._tool_properties,
            required_properties=self._tool_requirements,
        )

    def to_smolagents_tool(self) -> Any:
        """Converts the tool to a [smolagents tool](https://huggingface.co/docs/smolagents/reference/tools#smolagents.Tool).

        Note: You must have the [`smolagents` library installed](https://github.com/huggingface/smolagents) to use this method.
        """
        try:
            from cleanlab_codex.utils.smolagents import CodexTool as SmolagentsCodexTool
        except ImportError as e:
            raise MissingDependencyError(e.name or "smolagents", "https://github.com/huggingface/smolagents") from e

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
        try:
            from llama_index.core.tools import FunctionTool

        except ImportError as e:
            raise MissingDependencyError(
                import_name=e.name or "llama_index",
                package_name="llama-index-core",
                package_url="https://docs.llamaindex.ai/en/stable/getting_started/installation/",
            ) from e

        return FunctionTool.from_defaults(
            fn=self.query,
            name=self._tool_name,
            description=self._tool_description,
            fn_schema=self._tool_function_schema,
        )

    def to_langchain_tool(self) -> Any:
        """Converts the tool to a [LangChain tool](https://python.langchain.com/docs/concepts/tools/).

        Note: You must have the [`langchain` library installed](https://python.langchain.com/docs/concepts/architecture/) to use this method.
        """
        try:
            from langchain_core.tools.structured import StructuredTool

        except ImportError as e:
            raise MissingDependencyError(
                import_name=e.name or "langchain",
                package_name="langchain",
                package_url="https://pypi.org/project/langchain/",
            ) from e

        return StructuredTool.from_function(
            func=self.query,
            name=self._tool_name,
            description=self._tool_description,
            args_schema=self._tool_function_schema,
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
