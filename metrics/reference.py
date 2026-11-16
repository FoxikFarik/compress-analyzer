from models.claim import Claim
from models.reference_metrics_result import ReferenceMetricsResult
from retrieval.claim_matcher import ClaimMatcher


class ReferenceMetricsEvaluator:
    """Оценка конспекта относительно эталонных утверждений."""

    def __init__(self, claim_matcher: ClaimMatcher):
        self.claim_matcher = claim_matcher

    def evaluate(
        self,
        summary_claims: list[Claim],
        reference_claims: list[Claim]
    ) -> ReferenceMetricsResult:
        """Вычисляет Precision, Recall и F1."""

        matches = self.claim_matcher.match(
            summary_claims,
            reference_claims
        )

        tp = sum(
            1 for result in matches
            if result.matched
        )

        fp = len(summary_claims) - tp
        fn = len(reference_claims) - tp

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return ReferenceMetricsResult(
            precision=precision,
            recall=recall,
            f1=f1,
            matches=matches
        )
