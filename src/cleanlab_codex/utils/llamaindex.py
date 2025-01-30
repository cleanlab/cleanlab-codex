from __future__ import annotations

from inspect import signature
from typing import Any

from beartype.typing import Callable, Dict, Type
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo, create_model


def get_function_schema(name: str, func: Callable[..., Any], tool_properties: Dict[str, Any]) -> Type[BaseModel]:
    fields = {}
    params = signature(func).parameters
    for param_name in params:
        param_type = params[param_name].annotation
        param_default = params[param_name].default
        description = tool_properties.get(param_name, {}).get("description", None)

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            fields[param_name] = (param_type, FieldInfo(description=description))
        elif isinstance(param_default, FieldInfo):
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (
                param_type,
                FieldInfo(default=param_default, description=description),
            )

    return create_model(name, **fields)  # type: ignore
