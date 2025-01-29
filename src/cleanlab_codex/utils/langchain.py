from pydantic import BaseModel, Field
from typing import Any, Callable, Dict
from inspect import signature


def create_args_schema(
    name: str, func: Callable[..., Any], tool_properties: Dict[str, Any]
) -> type[BaseModel]:
    """
    Creates a pydantic BaseModel for langchain's args_schema.

    Args:
        name: Name of the schema.
        func: The function for which the schema is being generated.
        tool_properties: Metadata about each argument.

    Returns:
        type[BaseModel]: A pydantic model, annotated as required by langchain.
    """
    fields = {}
    params = signature(func).parameters

    for param_name, param in params.items():
        param_type = param.annotation if param.annotation is not param.empty else Any
        param_default = param.default
        description = tool_properties.get(param_name, {}).get("description", None)

        if param_default is param.empty:
            fields[param_name] = (param_type, Field(description=description))
        else:
            fields[param_name] = (
                param_type,
                Field(default=param_default, description=description),
            )

    return type(
        name,
        (BaseModel,),
        {
            "__annotations__": {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()},
        },
    )
