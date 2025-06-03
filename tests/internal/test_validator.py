import pytest

from cleanlab_codex.internal.validator import validate_thresholds


def test_validate_thresholds_type_error() -> None:
    """Test that validate_thresholds raises TypeError for non-numeric values."""
    with pytest.raises(TypeError, match="Threshold for my_eval must be a number, got <class 'str'>"):
        validate_thresholds({"my_eval": "not a number"})  # type: ignore[dict-item]


def test_validate_thresholds_value_error() -> None:
    """Test that validate_thresholds raises ValueError for values outside [0,1]."""
    with pytest.raises(ValueError, match="Threshold for my_eval must be between 0 and 1, got 1.5"):
        validate_thresholds({"my_eval": 1.5})


def test_validate_thresholds_success() -> None:
    """Test that validate_thresholds accepts valid threshold values."""
    validate_thresholds(
        {
            "eval1": 0.0,
            "eval2": 0.5,
            "eval3": 0.99,
        }
    )
