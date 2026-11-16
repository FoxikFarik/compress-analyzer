from models.evaluation_report import EvaluationReport
from models.faithfulness_report import FaithfulnessReport
from models.faithfulness_result import FaithfulnessResult
from models.reference_metrics_result import ReferenceMetricsResult


def test_evaluation_report_stores_basic_metrics():
    """Проверяет сохранение базовых метрик."""

    report = EvaluationReport(
        compression_ratio=0.35,
        duplicate_rate=0.1
    )

    assert report.compression_ratio == 0.35
    assert report.duplicate_rate == 0.1
    assert report.faithfulness is None


def test_evaluation_report_stores_faithfulness():
    """Проверяет сохранение отчёта Faithfulness."""

    faithfulness = FaithfulnessReport(
        sentence_results=[
            FaithfulnessResult(
                sentence="Проверяемое предложение."
            )
        ]
    )

    report = EvaluationReport(
        compression_ratio=0.4,
        duplicate_rate=0.05,
        faithfulness=faithfulness
    )

    assert report.faithfulness is faithfulness
    assert report.faithfulness.score == 0.0
    
def test_evaluation_report_with_reference_metrics():
    """Проверяет сохранение метрик сравнения с эталоном."""

    metrics = ReferenceMetricsResult(
        precision=0.8,
        recall=0.6,
        f1=0.685714
    )

    report = EvaluationReport(
        compression_ratio=0.3,
        duplicate_rate=0.1,
        reference_metrics=metrics
    )

    assert report.reference_metrics == metrics