from typing import Dict, List, Literal

from pydantic import BaseModel


class Property(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Property]
    required: List[str]
