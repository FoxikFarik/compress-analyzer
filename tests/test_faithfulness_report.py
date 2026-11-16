import pytest

from models.faithfulness_report import FaithfulnessReport
from models.faithfulness_result import (
    FaithfulnessResult,
    FragmentCheckResult
)
from models.fragment import Fragment
from models.nli_result import NLIResult


@pytest.fixture
def sentence_results():
    """Создаёт результаты проверки предложений."""

    return [
        FaithfulnessResult(
            sentence="Первое предложение.",
            fragment_results=[
                FragmentCheckResult(
                    fragment=Fragment(
                        text="Фрагмент 1",
                        paragraph_index=0,
                        start_sentence_index=0,
                        end_sentence_index=0,
                        score=0.9
                    ),
                    nli_result=NLIResult(0.8, 0.1, 0.1)
                )
            ]
        ),
        FaithfulnessResult(
            sentence="Второе предложение.",
            fragment_results=[
                FragmentCheckResult(
                    fragment=Fragment(
                        text="Фрагмент 2",
                        paragraph_index=0,
                        start_sentence_index=1,
                        end_sentence_index=1,
                        score=0.8
                    ),
                    nli_result=NLIResult(0.6, 0.2, 0.2)
                )
            ]
        )
    ]


def test_score_calculates_average(sentence_results):
    """Проверяет вычисление средней оценки."""

    report = FaithfulnessReport(
        sentence_results=sentence_results
    )

    assert report.score == pytest.approx(0.7)


def test_score_without_results():
    """Проверяет отчёт без результатов проверки."""

    report = FaithfulnessReport()

    assert report.score is None


def test_report_preserves_sentence_results(sentence_results):
    """Проверяет сохранение результатов проверки."""

    report = FaithfulnessReport(
        sentence_results=sentence_results
    )

    assert report.sentence_results == sentence_results
    
def create_fragment(score):
    """Создаёт тестовый фрагмент."""

    return Fragment(
        text="Тестовый фрагмент.",
        paragraph_index=0,
        start_sentence_index=0,
        end_sentence_index=0,
        score=score
    )


def test_faithfulness_result_without_fragments():
    """Проверяет оценку предложения без найденных фрагментов."""

    result = FaithfulnessResult(
        sentence="Тестовое предложение."
    )

    assert result.score == 0.0


def test_faithfulness_result_uses_max_entailment():
    """Проверяет выбор максимальной оценки Entailment."""

    result = FaithfulnessResult(
        sentence="Тестовое предложение.",
        fragment_results=[
            FragmentCheckResult(
                fragment=create_fragment(0.8),
                nli_result=NLIResult(0.5, 0.3, 0.2)
            ),
            FragmentCheckResult(
                fragment=create_fragment(0.6),
                nli_result=NLIResult(0.9, 0.05, 0.05)
            ),
            FragmentCheckResult(
                fragment=create_fragment(0.4),
                nli_result=NLIResult(0.7, 0.1, 0.2)
            )
        ]
    )

    assert result.score == 0.9


def test_faithfulness_report_without_results():
    """Проверяет пустой отчёт."""

    report = FaithfulnessReport()

    assert report.score is None


def test_faithfulness_report_calculates_average():
    """Проверяет усреднение оценок предложений."""

    report = FaithfulnessReport(
        sentence_results=[
            FaithfulnessResult(
                sentence="Первое предложение.",
                fragment_results=[
                    FragmentCheckResult(
                        fragment=create_fragment(0.8),
                        nli_result=NLIResult(0.8, 0.1, 0.1)
                    )
                ]
            ),
            FaithfulnessResult(
                sentence="Второе предложение.",
                fragment_results=[
                    FragmentCheckResult(
                        fragment=create_fragment(0.6),
                        nli_result=NLIResult(0.6, 0.2, 0.2)
                    )
                ]
            )
        ]
    )

    assert report.score == 0.7
