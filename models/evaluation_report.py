from dataclasses import dataclass

from models.faithfulness_report import FaithfulnessReport
from models.reference_metrics_result import ReferenceMetricsResult


@dataclass
class EvaluationReport:
    """Представляет общий отчёт об оценке конспекта."""

    compression_ratio: float
    duplicate_rate: float
    faithfulness: FaithfulnessReport | None = None
    reference_metrics: ReferenceMetricsResult | None = None
