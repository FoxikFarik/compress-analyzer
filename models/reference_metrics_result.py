from dataclasses import dataclass, field

from models.claim_match_result import ClaimMatchResult


@dataclass
class ReferenceMetricsResult:
    """Представляет результаты сравнения конспекта с эталоном."""

    precision: float
    recall: float
    f1: float
    matches: list[ClaimMatchResult] = field(default_factory=list)
