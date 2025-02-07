from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codex import Codex as _Codex

from cleanlab_codex.types.organization import Organization


def list_organizations(client: _Codex) -> list[Organization]:
    return [
        Organization.model_validate(org.model_dump()) for org in client.users.myself.organizations.list().organizations
    ]
