"""Types for Codex entries."""

from codex.types.projects.entry_create_params import EntryCreateParams
from codex.types.projects.entry_query_response import Entry as _Entry

from cleanlab_codex.internal.utils import generate_class_docstring


class EntryCreate(EntryCreateParams): ...


EntryCreate.__doc__ = f"""
Input type for creating a new Entry in a Codex project. Use this class to add a new Question-Answer pair to a project.

{generate_class_docstring(EntryCreateParams, name=EntryCreate.__name__)}
"""


class Entry(_Entry): ...


Entry.__doc__ = f"""
Type representing an Entry in a Codex project. This is the complete data structure returned from the Codex API, including system-generated fields like ID and timestamps.

{generate_class_docstring(_Entry, name=Entry.__name__)}
"""

__all__ = ["EntryCreate", "Entry"]
