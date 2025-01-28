from typing import Callable, Dict, Optional

from smolagents import Tool  # type: ignore


class CodexTool(Tool):  # type: ignore[misc]
    def __init__(
        self,
        query: Callable[[str], Optional[str]],
        tool_name: str,
        tool_description: str,
        inputs: Dict[str, Dict[str, str]],
    ):
        super().__init__()
        self._query = query
        self.name = tool_name
        self.description = tool_description
        self.inputs = inputs
        self.output_type = "string"

    def forward(self, question: str) -> Optional[str]:
        return self._query(question)
