from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Type

from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated, get_args, get_origin


class Property(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Property]
    required: List[str]


def pydantic_model_from_function(name: str, func: Callable[..., Any]) -> Type[BaseModel]:
    """
    Create a pydantic model representing a function's schema.

    For example, a function with the following signature:

    ```python
    def my_function(
        a: Annotated[int, "This is an integer"], b: str = "default"
    ) -> None: ...
    ```

    will be represented by the following pydantic model when `name="my_function"`:

    ```python
    class my_function(BaseModel):
        a: int = Field(description="This is an integer")
        b: str = "default"
    ```

    Args:
        name: The name for the pydantic model.
        func: The function to create a schema for.

    Returns:
        A pydantic model representing the function's schema.
    """
    fields = {}
    params = signature(func).parameters

    for param_name, param in params.items():
        param_type = param.annotation
        if isinstance(param_type, str):
            param_type = eval(param_type)  # noqa: S307

        param_default = param.default
        description = None

        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            param_type = args[0]
            if isinstance(args[1], str):
                description = args[1]

        if param_type is param.empty:
            param_type = Any

        if param_default is param.empty:
            fields[param_name] = (param_type, Field(description=description))
        else:
            fields[param_name] = (
                param_type,
                Field(default=param_default, description=description),
            )

    return create_model(name, **fields)  # type: ignore


def required_properties_from_model(model: Type[BaseModel]) -> List[str]:
    """Returns a list of required properties from a pydantic model."""
    return [name for name, field in model.model_fields.items() if field.is_required()]
