from unittest.mock import MagicMock

from llama_index.core.tools import FunctionTool

from cleanlab_codex.codex_tool import CodexTool


def test_to_openai_tool(mock_client: MagicMock):
    tool = CodexTool.from_access_key("")
    openai_tool = tool.to_openai_tool()
    assert openai_tool.get("type") == "function"
    assert openai_tool.get("function", {}).get("name") == tool.tool_name
    assert openai_tool.get("function", {}).get("description") == tool.tool_description
    assert openai_tool.get("function", {}).get("parameters", {}).get("type") == "object"
    assert (
        openai_tool.get("function", {}).get("parameters", {}).get("properties")
        == tool._tool_properties
    )
    assert (
        openai_tool.get("function", {}).get("parameters", {}).get("required")
        == tool._tool_requirements
    )


def test_to_llamaindex_tool(mock_client: MagicMock):
    tool = CodexTool.from_access_key("")
    llama_index_tool = tool.to_llamaindex_tool()
    assert isinstance(llama_index_tool, FunctionTool)
    assert llama_index_tool.metadata.name == tool.tool_name
    assert llama_index_tool.metadata.description == tool.tool_description
    assert llama_index_tool.fn == tool.query
