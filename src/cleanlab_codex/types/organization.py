"""Types for Cleanlab organizations. Codex is the API interface to the Cleanlab AI Platform."""

from codex.types.users.myself.user_organizations_schema import (
    Organization as _Organization,
)

from cleanlab_codex.internal.utils import generate_class_docstring


class Organization(_Organization): ...


Organization.__doc__ = f"""
Type representing a Cleanlab organization in the Cleanlab AI Platform.

{generate_class_docstring(_Organization, name=Organization.__name__)}
"""

__all__ = ["Organization"]
