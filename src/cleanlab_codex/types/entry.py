"""Types for Codex entries."""

from codex.types.projects.entry import Entry as _Entry
from codex.types.projects.entry_create_params import EntryCreateParams


class EntryCreate(EntryCreateParams):
    """
    Input type for creating a new Entry in a Codex project. Use this class to add a new Question-Answer pair to a project.

    ```python
    class EntryCreate:
        question: str
        answer: Optional[str] = None
    ```
    """


class Entry(_Entry):
    """
    Type representing an Entry in a Codex project. This is the complete data structure returned from the Codex API, including system-generated fields like ID and timestamps.

    ```python
    class Entry:
        id: str
        question: str
        answer: Optional[str] = None
        created_at: datetime
        answer_at: Optional[datetime] = None
    ```
    """


__all__ = ["EntryCreate", "Entry"]
