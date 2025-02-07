from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from cleanlab_codex.utils.function import FunctionParameters


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


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
            parameters=FunctionParameters(
                properties=tool_properties,
                required=required_properties,
            ),
        )
    ).model_dump()
