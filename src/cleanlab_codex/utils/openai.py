from __future__ import annotations

from typing import Any, Literal

from beartype.typing import Dict, List
from pydantic import BaseModel


class Property(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Property]
    required: List[str]


class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


def format_as_openai_tool(
    tool_name: str,
    tool_description: str,
    tool_properties: Dict[str, Any],
    required_properties: List[str],
) -> Dict[str, Any]:
    return Tool(
        function=Function(
            name=tool_name,
            description=tool_description,
            parameters=FunctionParameters(properties=tool_properties, required=required_properties),
        )
    ).model_dump()
