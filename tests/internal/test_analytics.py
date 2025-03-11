from __future__ import annotations

from cleanlab_codex.__about__ import __version__ as package_version
from cleanlab_codex.internal.analytics import IntegrationType, _AnalyticsMetadata


def test_analytics_metadata_to_headers_uses_defaults() -> None:
    metadata = _AnalyticsMetadata(integration_type=IntegrationType.BACKUP)

    assert metadata.to_headers() == {
        "X-Integration-Type": IntegrationType.BACKUP,
        "X-Source": "cleanlab-codex-python",
        "X-Client-Library-Version": package_version,
    }
