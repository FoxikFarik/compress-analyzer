from models.nli_result import NLIResult
from nli.base import NLIModel


class FakeNLIModel(NLIModel):
    """Тестовая модель NLI с заранее заданными оценками."""

    def predict(
        self,
        premise: str,
        hypothesis: str
    ) -> NLIResult:
        """Возвращает фиксированный результат проверки."""

        return NLIResult(
            entailment=0.9,
            contradiction=0.05,
            neutral=0.05
        )
