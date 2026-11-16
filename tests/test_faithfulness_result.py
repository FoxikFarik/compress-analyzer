import pytest

from models.fragment import Fragment
from models.faithfulness_result import (
    FaithfulnessResult,
    FragmentCheckResult
)
from models.nli_result import NLIResult


@pytest.fixture
def fragments():
    """Создаёт несколько тестовых фрагментов."""
    return [
        Fragment(
            text="Первый фрагмент.",
            paragraph_index=0,
            start_sentence_index=0,
            end_sentence_index=0,
            score=0.9
        ),
        Fragment(
            text="Второй фрагмент.",
            paragraph_index=0,
            start_sentence_index=1,
            end_sentence_index=1,
            score=0.7
        )
    ]


def test_score_returns_max_entailment(fragments):
    """Проверяет выбор максимальной оценки Entailment."""
    result = FaithfulnessResult(
        sentence="Проверяемое предложение.",
        fragment_results=[
            FragmentCheckResult(
                fragment=fragments[0],
                nli_result=NLIResult(0.6, 0.2, 0.2)
            ),
            FragmentCheckResult(
                fragment=fragments[1],
                nli_result=NLIResult(0.9, 0.05, 0.05)
            )
        ]
    )

    assert result.score == 0.9


def test_score_without_fragments():
    """Проверяет оценку при отсутствии найденных фрагментов."""
    result = FaithfulnessResult(
        sentence="Проверяемое предложение."
    )

    assert result.score == 0.0


def test_score_with_single_fragment(fragments):
    """Проверяет оценку при наличии одного фрагмента."""
    result = FaithfulnessResult(
        sentence="Проверяемое предложение.",
        fragment_results=[
            FragmentCheckResult(
                fragment=fragments[0],
                nli_result=NLIResult(0.75, 0.15, 0.1)
            )
        ]
    )

    assert result.score == 0.75


def test_fragment_results_are_preserved(fragments):
    """Проверяет сохранение результатов всех проверок."""
    checks = [
        FragmentCheckResult(
            fragment=fragments[0],
            nli_result=NLIResult(0.8, 0.1, 0.1)
        ),
        FragmentCheckResult(
            fragment=fragments[1],
            nli_result=NLIResult(0.5, 0.2, 0.3)
        )
    ]

    result = FaithfulnessResult(
        sentence="Проверяемое предложение.",
        fragment_results=checks
    )

    assert result.fragment_results == checks
    assert len(result.fragment_results) == 2


def test_score_is_calculated_dynamically(fragments):
    """Проверяет пересчёт оценки после изменения результатов."""
    result = FaithfulnessResult(
        sentence="Проверяемое предложение.",
        fragment_results=[
            FragmentCheckResult(
                fragment=fragments[0],
                nli_result=NLIResult(0.6, 0.2, 0.2)
            )
        ]
    )

    assert result.score == 0.6

    result.fragment_results.append(
        FragmentCheckResult(
            fragment=fragments[1],
            nli_result=NLIResult(0.95, 0.02, 0.03)
        )
    )

    assert result.score == 0.95