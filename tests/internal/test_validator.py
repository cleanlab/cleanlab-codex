from typing import cast

from cleanlab_tlm.utils.rag import TrustworthyRAGScore

from cleanlab_codex.internal.validator import get_default_evaluations
from cleanlab_codex.validator import BadResponseThresholds


def make_scores(trustworthiness: float, response_helpfulness: float) -> TrustworthyRAGScore:
    scores = {
        "trustworthiness": {
            "score": trustworthiness,
        },
        "response_helpfulness": {
            "score": response_helpfulness,
        },
    }
    return cast(TrustworthyRAGScore, scores)


def make_is_bad_response_config(trustworthiness: float, response_helpfulness: float) -> BadResponseThresholds:
    return BadResponseThresholds(
        trustworthiness=trustworthiness,
        response_helpfulness=response_helpfulness,
    )


def test_get_default_evaluations() -> None:
    assert {evaluation.name for evaluation in get_default_evaluations()} == {"response_helpfulness"}
