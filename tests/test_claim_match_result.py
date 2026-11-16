from models.claim import Claim
from models.claim_match_result import ClaimMatchResult


def test_claim_match_result_creation():
    """Проверяет создание результата сопоставления."""

    summary_claim = Claim(
        text="Вода замерзает при 0 °C.",
        index=0,
        sentence_index=0,
        paragraph_index=0
    )

    reference_claim = Claim(
        text="Температура замерзания воды составляет 0 °C.",
        index=0,
        sentence_index=0,
        paragraph_index=0
    )

    result = ClaimMatchResult(
        summary_claim=summary_claim,
        reference_claim=reference_claim,
        similarity=0.95,
        matched=True
    )

    assert result.summary_claim == summary_claim
    assert result.reference_claim == reference_claim
    assert result.similarity == 0.95
    assert result.matched is True


def test_claim_match_result_not_matched():
    """Проверяет сохранение результата без совпадения."""

    claim1 = Claim("Вода замерзает.", 0, 0, 0)
    claim2 = Claim("Вода кипит.", 0, 0, 0)

    result = ClaimMatchResult(
        summary_claim=claim1,
        reference_claim=claim2,
        similarity=0.2,
        matched=False
    )

    assert result.matched is False