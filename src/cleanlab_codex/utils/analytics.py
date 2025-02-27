from __future__ import annotations


class AnalyticsMetadata:
    def __init__(self, *, integration_type: str, source: str = "codex-python-sdk"):
        self._integration_type = integration_type
        self._source = source

    def to_headers(self) -> dict[str, str]:
        return {
            "X-Integration-Type": self._integration_type,
            "X-Source": self._source,
        }
