import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.tools.structured import StructuredTool
from llama_index.core.tools import FunctionTool

from cleanlab_codex.codex_tool import CodexTool


def test_to_openai_tool(mock_client: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("")
    openai_tool = tool.to_openai_tool()
    assert openai_tool.get("type") == "function"
    assert openai_tool.get("function", {}).get("name") == tool.tool_name
    assert openai_tool.get("function", {}).get("description") == tool.tool_description
    assert openai_tool.get("function", {}).get("parameters", {}).get("type") == "object"


def test_to_llamaindex_tool(mock_client: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("")
    llama_index_tool = tool.to_llamaindex_tool()
    assert isinstance(llama_index_tool, FunctionTool)
    assert llama_index_tool.metadata.name == tool.tool_name
    assert llama_index_tool.metadata.description == tool.tool_description
    assert llama_index_tool.fn == tool.query


def test_to_langchain_tool(mock_client: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("")
    langchain_tool = tool.to_langchain_tool()
    assert isinstance(langchain_tool, StructuredTool)
    assert callable(langchain_tool)
    assert hasattr(langchain_tool, "name")
    assert hasattr(langchain_tool, "description")
    assert langchain_tool.name == tool.tool_name, f"Expected tool name 'test_tool', got '{langchain_tool.name}'."
    assert (
        langchain_tool.description == tool.tool_description
    ), f"Expected description 'A tool for testing.', got '{langchain_tool.description}'."


def test_to_aws_converse_tool(mock_client: MagicMock) -> None:  # noqa: ARG001
    tool = CodexTool.from_access_key("")
    aws_converse_tool = tool.to_llamaindex_tool()
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
def test_to_smolagents_tool(mock_client: MagicMock) -> None:  # noqa: ARG001
    from smolagents import Tool  # type: ignore

    tool = CodexTool.from_access_key("")
    smolagents_tool = tool.to_smolagents_tool()
    assert isinstance(smolagents_tool, Tool)
    assert smolagents_tool.name == tool.tool_name
    assert smolagents_tool.description == tool.tool_description
