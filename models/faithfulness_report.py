from dataclasses import dataclass, field

from models.faithfulness_result import FaithfulnessResult


@dataclass
class FaithfulnessReport:
    """Представляет итоговый отчёт об оценке Faithfulness."""

    sentence_results: list[FaithfulnessResult] = field(
        default_factory=list
    )

    @property
    def score(self) -> float | None:
        """Возвращает среднюю оценку Faithfulness."""
        if not self.sentence_results:
            return None

        return sum(
            result.score
            for result in self.sentence_results
        ) / len(self.sentence_results)