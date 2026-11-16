import pytest

from unittest.mock import patch
from models.claim import Claim
from retrieval.claim_matcher import ClaimMatcher
from retrieval import claim_matcher as claim_matcher_module
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


@pytest.fixture
def summary_claims():
    return [
        Claim("Вода замерзает при 0 °C.", 0, 0, 0),
        Claim("Вода кипит при 100 °C.", 1, 1, 0)
    ]


@pytest.fixture
def reference_claims():
    return [
        Claim("Температура замерзания воды составляет 0 °C.", 0, 0, 0),
        Claim("Вода закипает при температуре 100 °C.", 1, 1, 0)
    ]


def test_match_returns_result_for_each_summary_claim(
    summary_claims,
    reference_claims
):
    """Проверяет наличие результата для каждого утверждения конспекта."""

    matcher = ClaimMatcher()

    results = matcher.match(summary_claims, reference_claims)

    assert len(results) == len(summary_claims)


def test_match_uses_claims_from_input():
    reference_claim = Claim(
        text="Температура замерзания воды составляет 0 °C.",
        index=0,
        sentence_index=0,
        paragraph_index=0
    )

    summary_claim = Claim(
        text="Температура замерзания воды составляет 0 °C.",
        index=0,
        sentence_index=0,
        paragraph_index=0
    )

    matcher = ClaimMatcher(threshold=0.5)

    results = matcher.match(
        [summary_claim],
        [reference_claim]
    )

    assert len(results) == 1
    assert results[0].matched is True
    assert results[0].summary_claim is summary_claim
    assert results[0].reference_claim is reference_claim


def test_match_similarity_is_in_valid_range(
    summary_claims,
    reference_claims
):
    """Проверяет диапазон оценок сходства."""

    matcher = ClaimMatcher()

    results = matcher.match(summary_claims, reference_claims)

    for result in results:
        assert 0 <= result.similarity <= 1


def test_match_respects_threshold(
    summary_claims,
    reference_claims
):
    """Проверяет применение порога сходства."""

    matcher = ClaimMatcher(threshold=1.0)

    results = matcher.match(summary_claims, reference_claims)

    for result in results:
        assert result.matched == (result.similarity >= 1.0)


def test_match_with_empty_summary(reference_claims):
    """Проверяет обработку пустого конспекта."""

    matcher = ClaimMatcher()

    results = matcher.match([], reference_claims)

    assert results == []


def test_match_with_empty_reference():
    summary_claims = [
        Claim(
            text="Машинное обучение использует алгоритмы.",
            index=0,
            sentence_index=0,
            paragraph_index=0
        )
    ]

    matcher = ClaimMatcher()

    results = matcher.match(summary_claims, [])

    assert len(results) == 1
    assert results[0].summary_claim is summary_claims[0]
    assert results[0].reference_claim is None
    assert results[0].matched is False
    assert results[0].similarity == 0.0


def test_match_with_both_empty():
    """Проверяет обработку двух пустых списков."""

    matcher = ClaimMatcher()

    results = matcher.match([], [])

    assert results == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_match_rejects_invalid_threshold(threshold):
    """Проверяет отклонение некорректного порога."""

    with pytest.raises(ValueError):
        ClaimMatcher(threshold=threshold)
        
def test_match_does_not_reuse_reference_claim():
    reference_claims = [
        Claim(
            text="Машинное обучение использует алгоритмы.",
            index=0,
            sentence_index=0,
            paragraph_index=0
        )
    ]

    summary_claims = [
        Claim(
            text="Машинное обучение использует алгоритмы.",
            index=0,
            sentence_index=0,
            paragraph_index=0
        ),
        Claim(
            text="Алгоритмы применяются в машинном обучении.",
            index=1,
            sentence_index=1,
            paragraph_index=0
        )
    ]

    matcher = ClaimMatcher(threshold=0.3)

    results = matcher.match(
        summary_claims,
        reference_claims
    )

    matched_results = [
        result for result in results
        if result.matched
    ]

    assert len(matched_results) == 1
    assert sum(result.matched for result in results) == 1
    
def test_match_finds_global_optimal_assignment():
    summary_claims = [
        Claim("Конспект 1", 0, 0, 0),
        Claim("Конспект 2", 1, 1, 0)
    ]

    reference_claims = [
        Claim("Эталон A", 0, 0, 0),
        Claim("Эталон B", 1, 1, 0)
    ]

    matcher = ClaimMatcher(threshold=0.5)

    # Подменяем матрицу сходства, чтобы проверить именно алгоритм
    # распределения, независимо от особенностей TF-IDF.
    similarities = np.array([
        [0.90, 0.80],
        [0.85, 0.10]
    ])

    matcher_similarity = cosine_similarity

    try:
        import retrieval.claim_matcher as claim_matcher_module

        claim_matcher_module.cosine_similarity = lambda *args: similarities

        results = matcher.match(
            summary_claims,
            reference_claims
        )
    finally:
        claim_matcher_module.cosine_similarity = matcher_similarity

    assert len(results) == 2

    assert results[0].reference_claim is reference_claims[1]
    assert results[1].reference_claim is reference_claims[0]

    assert all(result.matched for result in results)
    
def test_match_when_summary_has_more_claims():
    summary_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0),
        Claim("Земля вращается вокруг Солнца.", 1, 1, 0),
        Claim("Кислород необходим для дыхания.", 2, 2, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0),
        Claim("Земля вращается вокруг Солнца.", 1, 1, 0)
    ]

    matcher = ClaimMatcher(threshold=0.5)

    results = matcher.match(
        summary_claims,
        reference_claims
    )

    assert len(results) == 3
    assert sum(result.matched for result in results) == 2

    assert results[0].matched is True
    assert results[1].matched is True

    assert results[2].matched is False
    assert results[2].reference_claim is None

def test_match_when_reference_has_more_claims():
    summary_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0),
        Claim("Земля вращается вокруг Солнца.", 1, 1, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0),
        Claim("Земля вращается вокруг Солнца.", 1, 1, 0),
        Claim("Кислород необходим для дыхания.", 2, 2, 0)
    ]

    matcher = ClaimMatcher(threshold=0.5)

    results = matcher.match(
        summary_claims,
        reference_claims
    )

    assert len(results) == 2
    assert all(result.matched for result in results)

    matched_references = [
        result.reference_claim
        for result in results
    ]

    assert reference_claims[0] in matched_references
    assert reference_claims[1] in matched_references
    assert reference_claims[2] not in matched_references
    
def test_match_with_no_valid_tokens():
    summary_claims = [
        Claim("123", 0, 0, 0),
        Claim("!!!", 1, 1, 0)
    ]

    reference_claims = [
        Claim("456", 0, 0, 0),
        Claim("???", 1, 1, 0)
    ]

    matcher = ClaimMatcher()

    results = matcher.match(
        summary_claims,
        reference_claims
    )

    assert len(results) == 2

    for result in results:
        assert result.matched is False
        assert result.reference_claim is None
        assert result.similarity == 0.0
        
def test_match_with_some_invalid_tokens():
    summary_claims = [
        Claim("123 !!!", 0, 0, 0),
        Claim("Вода замерзает при нуле градусов.", 1, 1, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0)
    ]

    matcher = ClaimMatcher()

    results = matcher.match(
        summary_claims,
        reference_claims
    )

    assert results[0].matched is False
    assert results[0].reference_claim is None

    assert results[1].matched is True
    assert results[1].reference_claim is reference_claims[0]

def test_match_with_duplicate_summary_claims():
    summary_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0),
        Claim("Вода замерзает при нуле градусов.", 1, 1, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0)
    ]

    matcher = ClaimMatcher()

    results = matcher.match(summary_claims, reference_claims)

    assert sum(result.matched for result in results) == 1
    assert sum(not result.matched for result in results) == 1

def test_match_with_equal_similarities():
    summary_claims = [
        Claim("Утверждение первое.", 0, 0, 0),
        Claim("Утверждение второе.", 1, 1, 0)
    ]

    reference_claims = [
        Claim("Эталон первое.", 0, 0, 0),
        Claim("Эталон второе.", 1, 1, 0)
    ]

    similarities = np.array([
        [0.8, 0.8],
        [0.8, 0.8]
    ])

    matcher = ClaimMatcher(threshold=0.5)

    with patch.object(
        claim_matcher_module,
        "cosine_similarity",
        return_value=similarities
    ):
        results = matcher.match(summary_claims, reference_claims)

    assert all(result.matched for result in results)

    matched_references = [
        result.reference_claim
        for result in results
    ]

    assert len(set(id(claim) for claim in matched_references)) == 2
    
def test_match_respects_exact_threshold():
    summary_claims = [
        Claim("Вода замерзает.", 0, 0, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает.", 0, 0, 0)
    ]

    similarities = np.array([[0.5]])

    matcher = ClaimMatcher(threshold=0.5)

    with patch.object(
        claim_matcher_module,
        "cosine_similarity",
        return_value=similarities
    ):
        results = matcher.match(summary_claims, reference_claims)

    assert results[0].matched is True
    assert results[0].similarity == 0.5