from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, computed_field

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
    name: ResponseValidationMethod
    fails_check: bool
    score: Dict[str, float]
    metadata: Dict[str, Any]

    def __bool__(self) -> bool:
        return self.fails_check

    def __repr__(self) -> str:
        """Return a string representation of the SingleResponseValidationResult."""
        pass_or_fail = "Passed Check" if self.fails_check else "Failed Check"
        metadata_str = ", metadata=..." if self.metadata else ""
        return f"SingleResponseValidationResult(name={self.name}, {pass_or_fail}, score={self.score}{metadata_str})"


class AggregatedResponseValidationResult(BaseResponseValidationResult):
    name: AggregatedResponseValidationMethod
    results: List[SingleResponseValidationResult]

    @computed_field  # type: ignore
    @property
    def fails_check(self) -> bool:
        return any(result.fails_check for result in self.results)

    def __bool__(self) -> bool:
        return self.fails_check

    def __repr__(self) -> str:
        """Return a string representation of the AggregatedResponseValidationResult."""
        pass_or_fail = "Passed Check" if self.fails_check else "Failed Check"
        return f"AggregatedResponseValidationResult(name={self.name}, {pass_or_fail}, results={self.results})"
