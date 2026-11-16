import pytest

from models.nli_result import NLIResult
from nli.base import NLIModel


class MockNLIModel(NLIModel):
    """Тестовая реализация NLI-модели."""

    def predict(
        self,
        premise: str,
        hypothesis: str
    ) -> NLIResult:
        return NLIResult(
            entailment=0.8,
            contradiction=0.1,
            neutral=0.1
        )


def test_nli_result_label():
    """Проверяет определение категории с наибольшей оценкой."""
    result = NLIResult(
        entailment=0.8,
        contradiction=0.1,
        neutral=0.1
    )

    assert result.label == "entailment"


def test_nli_result_contradiction():
    """Проверяет определение противоречия."""
    result = NLIResult(
        entailment=0.1,
        contradiction=0.7,
        neutral=0.2
    )

    assert result.label == "contradiction"


def test_nli_result_neutral():
    """Проверяет определение нейтральной категории."""
    result = NLIResult(
        entailment=0.2,
        contradiction=0.1,
        neutral=0.7
    )

    assert result.label == "neutral"


def test_nli_model_is_abstract():
    """Проверяет, что базовую модель нельзя создать напрямую."""
    with pytest.raises(TypeError):
        NLIModel()


def test_mock_nli_model():
    """Проверяет работу реализации базового интерфейса."""
    model = MockNLIModel()

    result = model.predict(
        premise="Машинное обучение использует алгоритмы.",
        hypothesis="Алгоритмы применяются в машинном обучении."
    )

    assert isinstance(result, NLIResult)
    assert result.label == "entailment"