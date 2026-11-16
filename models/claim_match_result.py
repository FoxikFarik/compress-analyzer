from dataclasses import dataclass

from models.claim import Claim


@dataclass
class ClaimMatchResult:
    """Представляет результат сопоставления утверждений."""

    summary_claim: Claim
    reference_claim: Claim | None
    similarity: float
    matched: bool