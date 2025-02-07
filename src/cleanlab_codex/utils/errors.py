from __future__ import annotations


class MissingDependencyError(Exception):
    """Raised when a lazy import is missing."""

    def __init__(self, import_name: str, package_url: str | None = None) -> None:
        self.import_name = import_name
        self.package_url = package_url

    def __str__(self) -> str:
        message = f"Failed to import {self.import_name}. Please install the package using `pip install {self.import_name}` and try again."
        if self.package_url:
            message += f" For more information, see {self.package_url}."
        return message
