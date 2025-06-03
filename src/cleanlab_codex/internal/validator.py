from __future__ import annotations


def validate_thresholds(thresholds: dict[str, float]) -> None:
    """Validate that all threshold values are between 0 and 1.

    Args:
        thresholds: Dictionary mapping eval names to their threshold values.

    Raises:
        TypeError: If any threshold value is not a number.
        ValueError: If any threshold value is not between 0 and 1.
    """
    for eval_name, threshold in thresholds.items():
        if not isinstance(threshold, (int, float)):
            error_msg = f"Threshold for {eval_name} must be a number, got {type(threshold)}"
            raise TypeError(error_msg)
        if not 0 <= float(threshold) <= 1:
            error_msg = f"Threshold for {eval_name} must be between 0 and 1, got {threshold}"
            raise ValueError(error_msg)
