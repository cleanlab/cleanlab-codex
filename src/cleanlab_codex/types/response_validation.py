from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, computed_field

# Type aliases for validation scores
SingleScoreDict = Dict[str, float]
NestedScoreDict = OrderedDict[str, SingleScoreDict]

"""Type alias for validation scores.

Scores can be either a single score or a nested dictionary of scores.

Example:
    # Single score
    scores: ValidationScores = {"score": 0.5}
    # Nested scores
    scores: ValidationScores = {
        "check_a": {"sub_score_a1": 0.5, "sub_score_a2": 0.5},
        "check_b": {"sub_score_b1": 0.5, "sub_score_b2": 0.5},
    }
"""
ValidationScores = Union[SingleScoreDict, NestedScoreDict]


ResponseValidationMethod = Literal["fallback", "untrustworthy", "unhelpful"]
AggregatedResponseValidationMethod = Literal["bad"]


class BaseResponseValidationResult(BaseModel, ABC):
    name: Union[ResponseValidationMethod, AggregatedResponseValidationMethod]

    @abstractmethod
    def __bool__(self) -> bool:
        raise NotImplementedError


class SingleResponseValidationResult(BaseResponseValidationResult):
    """Result of a single response validation check.

    This class represents the outcome of an individual validation check performed
    on an AI response.
    """

    name: ResponseValidationMethod = Field(description="The name of the validation check.")
    fails_check: bool = Field(description="Whether the check failed. True if the check failed, False otherwise.")
    score: Dict[str, float] = Field(
        description="The score of the response. Typically a single score's value is between 0.0 and 1.0, but this can vary by check."
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the response. This can include the threshold values, or other information relevant to the check."
    )

    def __bool__(self) -> bool:
        return self.fails_check

    def __repr__(self) -> str:
        pass_or_fail = "Failed Check" if self.fails_check else "Passed Check"
        metadata_str = ", metadata=..." if self.metadata else ""
        return f"SingleResponseValidationResult(name={self.name}, {pass_or_fail}, score={self.score}{metadata_str})"


class AggregatedResponseValidationResult(BaseResponseValidationResult):
    """Result of multiple combined response validation checks.

    This class aggregates multiple SingleResponseValidationResults and provides
    a combined validation outcome.

    The class is typically used in a boolean context to determine if any of the
    underlying checks failed. But each of the individual results are also
    accessible, via the `results` field.
    """

    name: AggregatedResponseValidationMethod = Field(description="The name of the aggregated validation check.")
    results: List[SingleResponseValidationResult] = Field(
        description="The individual results of the validation checks."
    )

    @computed_field  # type: ignore
    @property
    def fails_check(self) -> bool:
        """Whether any of the underlying checks failed."""
        return any(result.fails_check for result in self.results)

    def __bool__(self) -> bool:
        return self.fails_check

    def __repr__(self) -> str:
        pass_or_fail = "Passed Check" if self.fails_check else "Failed Check"
        return f"AggregatedResponseValidationResult(name={self.name}, {pass_or_fail}, results={self.results})"
