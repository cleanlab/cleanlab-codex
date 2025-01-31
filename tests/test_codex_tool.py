import sys
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.tools import FunctionTool

from cleanlab_codex.codex_tool import CodexTool


def test_to_openai_tool(mock_client: MagicMock) -> None:
    with patch("cleanlab_codex.codex_tool.Project") as mock_project:
        mock_project.from_access_key.return_value = MagicMock(client=mock_client, id="test_project_id")

        tool = CodexTool.from_access_key("sk-test-123")
        openai_tool = tool.to_openai_tool()
        assert openai_tool.get("type") == "function"
        assert openai_tool.get("function", {}).get("name") == tool.tool_name
        assert openai_tool.get("function", {}).get("description") == tool.tool_description
        assert openai_tool.get("function", {}).get("parameters", {}).get("type") == "object"


def test_to_llamaindex_tool(mock_client: MagicMock) -> None:
    with patch("cleanlab_codex.codex_tool.Project") as mock_project:
        mock_project.from_access_key.return_value = MagicMock(client=mock_client, id="test_project_id")

        tool = CodexTool.from_access_key("sk-test-123")
        llama_index_tool = tool.to_llamaindex_tool()
        assert isinstance(llama_index_tool, FunctionTool)
        assert llama_index_tool.metadata.name == tool.tool_name
        assert llama_index_tool.metadata.description == tool.tool_description
        assert llama_index_tool.fn == tool.query


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_to_smolagents_tool(mock_client: MagicMock) -> None:
    from smolagents import Tool  # type: ignore

    with patch("cleanlab_codex.codex_tool.Project") as mock_project:
        mock_project.from_access_key.return_value = MagicMock(client=mock_client, id="test_project_id")

        tool = CodexTool.from_access_key("sk-test-123")
        smolagents_tool = tool.to_smolagents_tool()
        assert isinstance(smolagents_tool, Tool)
        assert smolagents_tool.name == tool.tool_name
        assert smolagents_tool.description == tool.tool_description
