import builtins
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools.structured import StructuredTool
from llama_index.core.tools import FunctionTool

from cleanlab_codex.codex_tool import CodexTool
from cleanlab_codex.internal.analytics import IntegrationType, _AnalyticsMetadata
from cleanlab_codex.utils.errors import MissingDependencyError


def test_tool_query_passes_metadata(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    mock_project = MagicMock()
    mock_project.query.return_value = (None, None)

    with patch("cleanlab_codex.codex_tool.Project.from_access_key", return_value=mock_project):
        tool = CodexTool.from_access_key("sk-test-123")
        tool.query("what is the capital of France?")

        assert mock_project.query.call_count == 1
        args, kwargs = mock_project.query.call_args
        assert kwargs["question"] == "what is the capital of France?"
        assert kwargs["fallback_answer"] == CodexTool.DEFAULT_FALLBACK_ANSWER
        assert (
            kwargs["_analytics_metadata"].to_headers()
            == _AnalyticsMetadata(integration_type=IntegrationType.TOOL).to_headers()
        )


def patch_import_with_import_error(missing_module: str) -> None:
    def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(missing_module):
            raise ImportError("test", name=missing_module)
        return importlib.__import__(name, *args, **kwargs)

    builtins.__import__ = custom_import


def test_to_openai_tool(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("sk-test-123")
    openai_tool = tool.to_openai_tool()
    assert openai_tool.get("type") == "function"
    assert openai_tool.get("function", {}).get("name") == tool.tool_name
    assert openai_tool.get("function", {}).get("description") == tool.tool_description
    assert openai_tool.get("function", {}).get("parameters", {}).get("type") == "object"


def test_to_llamaindex_tool(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("sk-test-123")
    llama_index_tool = tool.to_llamaindex_tool()
    assert isinstance(llama_index_tool, FunctionTool)
    assert llama_index_tool.metadata.name == tool.tool_name
    assert llama_index_tool.metadata.description == tool.tool_description
    assert llama_index_tool.fn == tool.query


def test_to_llamaindex_tool_import_error(
    mock_client_from_access_key: MagicMock,  # noqa: ARG001
) -> None:
    tool = CodexTool.from_access_key("sk-test-123")
    patch_import_with_import_error("llama_index")
    with pytest.raises(MissingDependencyError) as exc_info:
        tool.to_llamaindex_tool()

    assert exc_info.value.import_name == "llama_index"


def test_to_langchain_tool(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("sk-test-123")
    langchain_tool = tool.to_langchain_tool()
    assert isinstance(langchain_tool, StructuredTool)
    assert callable(langchain_tool)
    assert hasattr(langchain_tool, "name")
    assert hasattr(langchain_tool, "description")
    assert langchain_tool.name == tool.tool_name, f"Expected tool name '{tool.tool_name}', got '{langchain_tool.name}'."
    assert (
        langchain_tool.description == tool.tool_description
    ), f"Expected description '{tool.tool_description}', got '{langchain_tool.description}'."


def test_to_langchain_tool_import_error(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("sk-test-123")
    patch_import_with_import_error("langchain")
    with pytest.raises(MissingDependencyError) as exc_info:
        tool.to_langchain_tool()

    assert exc_info.value.import_name == "langchain"


def test_to_aws_converse_tool(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("sk-test-123")
    aws_converse_tool = tool.to_aws_converse_tool()

    assert "toolSpec" in aws_converse_tool
    assert (
        aws_converse_tool["toolSpec"].get("name") == tool.tool_name
    ), f"Expected name '{tool.tool_name}', got '{aws_converse_tool['toolSpec'].get('name')}'"
    assert (
        aws_converse_tool["toolSpec"].get("description") == tool.tool_description
    ), f"Expected description '{tool.tool_description}', got '{aws_converse_tool['toolSpec'].get('description')}'"
    assert "inputSchema" in aws_converse_tool["toolSpec"], "inputSchema key is missing in toolSpec"

    input_schema = aws_converse_tool["toolSpec"]["inputSchema"]
    assert "json" in input_schema

    json_schema = input_schema["json"]
    assert json_schema.get("type") == "object"
    assert "properties" in json_schema

    properties = json_schema["properties"]
    assert "question" in properties

    question_property = properties["question"]
    assert question_property.get("type") == "string"
    assert "description" in question_property
    assert "required" in json_schema
    assert "question" in json_schema["required"]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_to_smolagents_tool(mock_client_from_access_key: MagicMock) -> None:  # noqa: ARG001
    from smolagents import Tool  # type: ignore

    tool = CodexTool.from_access_key("sk-test-123")
    smolagents_tool = tool.to_smolagents_tool()
    assert isinstance(smolagents_tool, Tool)
    assert smolagents_tool.name == tool.tool_name
    assert smolagents_tool.description == tool.tool_description


def test_to_smolagents_tool_import_error(
    mock_client_from_access_key: MagicMock,  # noqa: ARG001
) -> None:
    tool = CodexTool.from_access_key("sk-test-123")
    import_module_name = "smolagents"
    if sys.version_info >= (3, 10):
        import_module_name = "cleanlab_codex.utils.smolagents"
        patch_import_with_import_error(import_module_name)

    with pytest.raises(MissingDependencyError) as exc_info:
        tool.to_smolagents_tool()

    assert exc_info.value.import_name == import_module_name
