from __future__ import annotations

from cleanlab_codex.__about__ import __version__ as package_version


class AnalyticsMetadata:
    def __init__(
        self, *, integration_type: str, package_version: str = package_version, source: str = "cleanlab-codex-python"
    ):
        self._integration_type = integration_type
        self._package_version = package_version
        self._source = source

    def to_headers(self) -> dict[str, str]:
        return {
            "X-Integration-Type": self._integration_type,
            "X-Client-Library-Version": self._package_version,
            "X-Source": self._source,
        }
