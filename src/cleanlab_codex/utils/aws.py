from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from cleanlab_codex.utils.function import FunctionParameters


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, FunctionParameters] = Field(alias="inputSchema")


class Tool(BaseModel):
    tool_spec: ToolSpec = Field(alias="toolSpec")


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
            inputSchema={
                "json": FunctionParameters(
                    properties=tool_properties,
                    required=required_properties,
                )
            },
        )
    ).model_dump(by_alias=True)
