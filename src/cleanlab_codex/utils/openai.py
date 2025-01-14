from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Property(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Property]
    required: list[str]


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
    tool_properties: dict[str, Any],
    required_properties: list[str],
) -> dict[str, Any]:
    return Tool(
        function=Function(
            name=tool_name,
            description=tool_description,
            parameters=FunctionParameters(properties=tool_properties, required=required_properties),
        )
    ).model_dump()
