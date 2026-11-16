from models.claim import Claim


def test_claim_creation():
    """Проверяет создание утверждения."""
    claim = Claim(
        text="Вода замерзает при 0 °C.",
        index=0,
        sentence_index=2,
        paragraph_index=1
    )

    assert claim.text == "Вода замерзает при 0 °C."
    assert claim.index == 0
    assert claim.sentence_index == 2
    assert claim.paragraph_index == 1


def test_claim_equality():
    """Проверяет сравнение двух одинаковых утверждений."""
    claim1 = Claim(
        text="Вода замерзает при 0 °C.",
        index=0,
        sentence_index=2,
        paragraph_index=1
    )

    claim2 = Claim(
        text="Вода замерзает при 0 °C.",
        index=0,
        sentence_index=2,
        paragraph_index=1
    )

    assert claim1 == claim2


def test_claims_are_independent():
    """Проверяет независимость экземпляров модели."""
    claim1 = Claim(
        text="Первое утверждение.",
        index=0,
        sentence_index=0,
        paragraph_index=0
    )

    claim2 = Claim(
        text="Второе утверждение.",
        index=1,
        sentence_index=0,
        paragraph_index=0
    )

    assert claim1.text != claim2.text
    assert claim1.index != claim2.index