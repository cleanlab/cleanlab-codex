from __future__ import annotations

import pytest

from cleanlab_codex.__about__ import __version__ as package_version
from cleanlab_codex.utils.analytics import AnalyticsMetadata


def test_analytics_metadata_to_headers_uses_defaults() -> None:
    metadata = AnalyticsMetadata(integration_type="backup")

    assert metadata.to_headers() == {
        "X-Integration-Type": "backup",
        "X-Source": "cleanlab-codex-python",
        "X-Client-Library-Version": package_version,
    }


def test_analytics_metadata_to_headers_uses_custom_values() -> None:
    metadata = AnalyticsMetadata(integration_type="tool-call", source="test", package_version="2.0.0")
    assert metadata.to_headers() == {
        "X-Integration-Type": "tool-call",
        "X-Source": "test",
        "X-Client-Library-Version": "2.0.0",
    }


def test_analytics_metadata_requires_integration_type() -> None:
    with pytest.raises(TypeError):
        AnalyticsMetadata()  # type: ignore[call-arg]
