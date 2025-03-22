from cleanlab_tlm.utils.rag import EvalMetric


class ThresholdedEvalMetric(EvalMetric):
    is_bad: bool


ThresholdedEvalMetric.__doc__ = f"""
{EvalMetric.__doc__}

is_bad: bool
    Whether the score is a certain threshold.
"""


class ThresholdedTrustworthyRAGScore(dict[str, ThresholdedEvalMetric]):
    """Object returned by `Validator.detect` containing evaluation scores from [TrustworthyRAGScore](/tlm/api/python/utils.rag/#class-trustworthyragscore)
    along with a boolean flag, `is_bad`, indicating whether the score is below the threshold.

    Example:
        ```python
        {
            "trustworthiness": {
                "score": 0.92,
                "log": {"explanation": "Did not find a reason to doubt trustworthiness."},
                "is_bad": False
            },
            "response_helpfulness": {
                "score": 0.35,
                "is_bad": True
            },
            ...
        }
        ```
    """
