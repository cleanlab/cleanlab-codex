from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from pydantic_core import PydanticUndefinedType
from typing_extensions import get_origin, get_type_hints, is_typeddict

if _TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo


def generate_class_docstring(cls: type, name: str | None = None) -> str:
    if is_typeddict(cls):
        return docstring_from_type_hints(cls, name)

    return docstring_from_annotations(cls, name)


def docstring_from_type_hints(cls: type, name: str | None = None) -> str:
    formatted_type_hints = "\n    ".join(f"{k}: '{annotation_to_str(v)}'" for k, v in get_type_hints(cls).items())
    return f"""
```python
class {name or cls.__name__}{is_typeddict(cls) and "(TypedDict)"}:
    {formatted_type_hints}
```
"""


def docstring_from_annotations(cls: type, name: str | None = None) -> str:
    formatted_annotations = "\n    ".join(f"{k}: '{annotation_to_str(v)}'" for k, v in cls.__annotations__.items())
    return f"""
```python
class {name or cls.__name__}:
    {formatted_annotations}
```
"""


def generate_pydantic_model_docstring(cls: type[BaseModel], name: str) -> str:  # no cov  # TODO: unignore once used
    formatted_annotations = "\n    ".join(
        format_annotation_from_field_info(field_name, field_info) for field_name, field_info in cls.model_fields.items()
    )
    formatted_fields = "\n".join(
        format_pydantic_field_docstring(field_name, field_info) for field_name, field_info in cls.model_fields.items()
    )
    return f"""
```python
class {name}(BaseModel):
    {formatted_annotations}
```

**Args:**

{formatted_fields}
"""


def format_annotation_from_field_info(
    field_name: str, field_info: FieldInfo
) -> str:  # no cov  # TODO: unignore once used
    annotation = field_name
    if field_info.annotation:
        annotation += f": '{annotation_to_str(field_info.annotation)}'"
    if field_info.default and not isinstance(field_info.default, PydanticUndefinedType):
        annotation += f" = {field_info.default}"
    return annotation


def format_pydantic_field_docstring(
    field_name: str, field_info: FieldInfo
) -> str:  # no cov  # TODO: unignore once used
    arg_str = f"- **`{field_name}`**"
    if field_info.annotation:
        arg_str += f" ({annotation_to_str(field_info.annotation)})"
    if field_info.description:
        arg_str += f": {field_info.description}"
    return arg_str


def annotation_to_str(annotation: type) -> str:
    if isinstance(annotation, str):
        return f"{annotation}"
    if get_origin(annotation) is None:
        return f"{annotation.__name__}"
    return f"{annotation!r}"
