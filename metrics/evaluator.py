from models.claim import Claim
from models.document import Document
from models.evaluation_report import EvaluationReport

from metrics.basic import (
    calculate_compression_ratio,
    calculate_duplicate_rate
)
from metrics.faithfulness import FaithfulnessEvaluator
from metrics.reference import ReferenceMetricsEvaluator


class DocumentEvaluator:
    """Общий координатор оценки качества конспекта."""

    def __init__(
        self,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
        reference_metrics_evaluator: ReferenceMetricsEvaluator | None = None
    ):
        self.faithfulness_evaluator = faithfulness_evaluator
        self.reference_metrics_evaluator = reference_metrics_evaluator

    def evaluate(
        self,
        source: Document,
        summary: Document,
        summary_claims: list[Claim] | None = None,
        reference_claims: list[Claim] | None = None
    ) -> EvaluationReport:
        """Выполняет комплексную оценку качества конспекта."""

        compression_ratio = calculate_compression_ratio(
            source,
            summary
        )

        duplicate_rate = calculate_duplicate_rate(summary)

        faithfulness = None

        if self.faithfulness_evaluator is not None:
            faithfulness = self.faithfulness_evaluator.evaluate_report(
                summary
            )

        reference_metrics = None

        if self.reference_metrics_evaluator is not None:
            if summary_claims is None or reference_claims is None:
                raise ValueError(
                    "Для расчёта Reference Metrics необходимо "
                    "передать утверждения конспекта и эталона"
                )

            reference_metrics = self.reference_metrics_evaluator.evaluate(
                summary_claims,
                reference_claims
            )

        return EvaluationReport(
            compression_ratio=compression_ratio,
            duplicate_rate=duplicate_rate,
            faithfulness=faithfulness,
            reference_metrics=reference_metrics
        )
