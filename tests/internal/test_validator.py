from typing import cast

import pytest
from cleanlab_tlm.utils.rag import TrustworthyRAGScore

from cleanlab_codex.internal.validator import BadResponseThresholds, get_default_evaluations, is_bad_response


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


class TestIsBadResponse:
    @pytest.fixture
    def scores(self) -> TrustworthyRAGScore:
        return make_scores(0.92, 0.75)

    @pytest.fixture
    def custom_is_bad_response_config(self) -> BadResponseThresholds:
        return make_is_bad_response_config(0.6, 0.7)

    def test_thresholds(self, scores: TrustworthyRAGScore) -> None:
        default_is_bad_response = is_bad_response(scores)
        assert not default_is_bad_response

        # High trustworthiness_threshold
        is_bad_response_config = make_is_bad_response_config(0.921, 0.5)
        assert is_bad_response(scores, is_bad_response_config)

        # High response_helpfulness_threshold
        is_bad_response_config = make_is_bad_response_config(0.5, 0.751)
        assert is_bad_response(scores, is_bad_response_config)

    def test_scores(self, custom_is_bad_response_config: BadResponseThresholds) -> None:
        scores = make_scores(0.59, 0.7)
        assert is_bad_response(scores, custom_is_bad_response_config)
