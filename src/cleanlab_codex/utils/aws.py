from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from cleanlab_codex.utils.types import FunctionParameters


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParameters]


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


def format_as_aws_converse_tool(
    tool_name: str,
    tool_description: str,
    input_properties: Dict[str, Any],
    required_properties: List[str],
) -> Dict[str, Any]:
    return Tool(
        function=Function(
            name=tool_name,
            description=tool_description,
            parameters={
                "json": FunctionParameters(properties=input_properties, required=required_properties)
            },
        )
    ).model_dump()

