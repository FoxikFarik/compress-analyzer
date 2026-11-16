from dataclasses import dataclass


@dataclass
class NLIResult:
    """Представляет результат проверки логического следования."""

    entailment: float
    contradiction: float
    neutral: float

    @property
    def label(self) -> str:
        """Возвращает категорию с наибольшей оценкой."""
        scores = {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral
        }

        return max(scores, key=scores.get)