from __future__ import annotations

from enum import Enum

from cleanlab_codex.__about__ import __version__ as package_version


class IntegrationType(str, Enum):
    """Supported methods for integrating Codex into a RAG system using this library."""

    BACKUP = "backup"


class _AnalyticsMetadata:
    def __init__(self, *, integration_type: IntegrationType | None = None):
        self._integration_type = integration_type
        self._package_version = package_version
        self._source = "cleanlab-codex-python"

    def to_headers(self) -> dict[str, str]:
        return {
            "X-Integration-Type": self._integration_type or "None",
            "X-Client-Library-Version": self._package_version,
            "X-Source": self._source,
        }
