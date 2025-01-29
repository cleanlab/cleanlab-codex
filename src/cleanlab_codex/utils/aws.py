from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from cleanlab_codex.utils.types import FunctionParameters


class ToolSpec(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, FunctionParameters]


class Tool(BaseModel):
    toolSpec: ToolSpec


def format_as_aws_converse_tool(
    tool_name: str,
    tool_description: str,
    tool_properties: Dict[str, Any],
    required_properties: List[str],
) -> Dict[str, Any]:
    return Tool(
        function=ToolSpec(
            name=tool_name,
            description=tool_description,
            parameters={
                "json": FunctionParameters(properties=tool_properties, required=required_properties)
            },
        )
    ).model_dump()

