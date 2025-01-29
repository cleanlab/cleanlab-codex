from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from cleanlab_codex.utils.types import FunctionParameters


class ToolSpec(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, FunctionParameters]  # noqa: N815


class Tool(BaseModel):
    toolSpec: ToolSpec  # noqa: N815


def format_as_aws_converse_tool(
    tool_name: str,
    tool_description: str,
    tool_properties: Dict[str, Any],
    required_properties: List[str],
) -> Dict[str, Any]:
    return Tool(
        toolSpec=ToolSpec(
            name=tool_name,
            description=tool_description,
            inputSchema={"json": FunctionParameters(properties=tool_properties, required=required_properties)},
        )
    ).model_dump()
