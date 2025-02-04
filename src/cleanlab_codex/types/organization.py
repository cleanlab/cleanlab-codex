"""Types for Codex organizations."""

from codex.types.users.myself.user_organizations_schema import Organization as _Organization


class Organization(_Organization):
    """
    Type representing an organization in Codex.

    ```python
    class Organization:
        id: str
        name: str
        payment_status: Literal[
            "NULL", "FIRST_OVERAGE_LENIENT", "SECOND_OVERAGE_USAGE_BLOCKED"
        ]
        created_at: datetime
        updated_at: datetime
    ```
    """


__all__ = ["Organization"]
