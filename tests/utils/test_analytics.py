from cleanlab_codex.utils.analytics import AnalyticsMetadata


def test_analytics_metadata_to_headers_uses_default_source() -> None:
    metadata = AnalyticsMetadata(integration_type="backup")
    assert metadata.to_headers() == {"X-Integration-Type": "backup", "X-Source": "codex-python-sdk"}


def test_analytics_metadata_to_headers_uses_integration_type() -> None:
    metadata = AnalyticsMetadata(integration_type="tool-call", source="test")
    assert metadata.to_headers() == {"X-Integration-Type": "tool-call", "X-Source": "test"}


def test_analytics_metadata_to_headers_uses_integration_type() -> None:
    try:
        metadata = AnalyticsMetadata()
    except TypeError:
        pass
    else:
        raise AssertionError("Expected a TypeError due to missing required argument")
