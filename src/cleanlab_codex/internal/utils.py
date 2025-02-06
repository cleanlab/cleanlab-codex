from __future__ import annotations

from typing_extensions import get_origin, get_type_hints, is_typeddict


def generate_class_docstring(cls: type, name: str | None = None) -> str:
    if is_typeddict(cls):
        return docstring_from_type_hints(cls, name)

    return docstring_from_annotations(cls, name)


def docstring_from_type_hints(cls: type, name: str | None = None) -> str:
    formatted_type_hints = "\n    ".join(f"{k}: {annotation_to_str(v)}" for k, v in get_type_hints(cls).items())
    return f"""
```python
class {name or cls.__name__}{is_typeddict(cls) and "(TypedDict)"}:
    {formatted_type_hints}
```
"""


def docstring_from_annotations(cls: type, name: str | None = None) -> str:
    formatted_annotations = "\n    ".join(f"{k}: {annotation_to_str(v)}" for k, v in cls.__annotations__.items())
    return f"""
```python
class {name or cls.__name__}:
    {formatted_annotations}
```
"""


def annotation_to_str(annotation: type) -> str:
    if get_origin(annotation) is None:
        return annotation.__name__
    return repr(annotation)
