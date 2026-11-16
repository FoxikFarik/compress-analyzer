import pytest

from models.claim import Claim
from retrieval.claim_matcher import ClaimMatcher
from metrics.reference import ReferenceMetricsEvaluator


@pytest.fixture
def evaluator():
    """Создаёт evaluator с обычным порогом сходства."""
    return ReferenceMetricsEvaluator(
        claim_matcher=ClaimMatcher(threshold=0.5)
    )


@pytest.fixture
def reference_claims():
    """Создаёт набор эталонных утверждений."""
    return [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0),
        Claim("Вода кипит при 100 °C.", 1, 1, 0),
        Claim("Вода расширяется при замерзании.", 2, 2, 0)
    ]


def test_perfect_match(evaluator):
    """Проверяет оценку при полном совпадении."""

    claims = [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0),
        Claim("Вода кипит при 100 °C.", 1, 1, 0)
    ]

    result = evaluator.evaluate(claims, claims)

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_partial_match(evaluator, reference_claims):
    """Проверяет оценку при частичном совпадении."""

    summary_claims = [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0),
        Claim("Вода кипит при 100 °C.", 1, 1, 0),
        Claim("Лёд имеет меньшую плотность, чем вода.", 2, 2, 0)
    ]

    result = evaluator.evaluate(
        summary_claims,
        reference_claims
    )

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_no_matches(evaluator):
    """Проверяет оценку при отсутствии совпадений."""

    summary_claims = [
        Claim("Солнце является звездой.", 0, 0, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0)
    ]

    result = evaluator.evaluate(
        summary_claims,
        reference_claims
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_empty_summary(evaluator, reference_claims):
    """Проверяет обработку пустого конспекта."""

    result = evaluator.evaluate([], reference_claims)

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_empty_reference(evaluator):
    """Проверяет обработку пустого эталона."""

    summary_claims = [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0)
    ]

    result = evaluator.evaluate(summary_claims, [])

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_both_empty(evaluator):
    """Проверяет обработку двух пустых наборов."""

    result = evaluator.evaluate([], [])

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
