from models.reference_metrics_result import ReferenceMetricsResult


def test_reference_metrics_result_creation():
    """Проверяет создание результата оценки."""

    result = ReferenceMetricsResult(
        precision=0.8,
        recall=0.6,
        f1=0.685714
    )

    assert result.precision == 0.8
    assert result.recall == 0.6
    assert result.f1 == 0.685714


def test_reference_metrics_result_equality():
    """Проверяет сравнение двух одинаковых результатов."""

    result1 = ReferenceMetricsResult(
        precision=0.8,
        recall=0.6,
        f1=0.685714
    )

    result2 = ReferenceMetricsResult(
        precision=0.8,
        recall=0.6,
        f1=0.685714
    )

    assert result1 == result2


def test_reference_metrics_results_are_different():
    """Проверяет сравнение разных результатов."""

    result1 = ReferenceMetricsResult(
        precision=0.8,
        recall=0.6,
        f1=0.685714
    )

    result2 = ReferenceMetricsResult(
        precision=0.9,
        recall=0.6,
        f1=0.72
    )

    assert result1 != result2
