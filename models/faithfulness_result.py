from dataclasses import dataclass, field

from models.fragment import Fragment
from models.nli_result import NLIResult


@dataclass
class FragmentCheckResult:
    """Представляет результат проверки одного найденного фрагмента."""

    fragment: Fragment
    nli_result: NLIResult


@dataclass
class FaithfulnessResult:
    """Представляет результат оценки одного предложения конспекта."""

    sentence: str
    fragment_results: list[FragmentCheckResult] = field(
        default_factory=list
    )

    @property
    def score(self) -> float:
        """Возвращает максимальную оценку логического следования."""
        if not self.fragment_results:
            return 0.0

        return max(
            result.nli_result.entailment
            for result in self.fragment_results
        )