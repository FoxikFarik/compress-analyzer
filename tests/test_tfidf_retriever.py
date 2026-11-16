import pytest

from retrieval.tfidf import TfidfRetriever
from models.retrieval_result import RetrievalResult


@pytest.fixture
def retriever():
    """Создаёт подготовленный TF-IDF-поисковик."""
    instance = TfidfRetriever()
    instance.fit([
        "Машинное обучение изучает алгоритмы.",
        "Нейронные сети применяются в машинном обучении.",
        "Python используется для программирования."
    ])
    return instance


def test_fit_empty_documents():
    """Проверяет запрет на пустой список документов."""
    retriever = TfidfRetriever()

    with pytest.raises(ValueError):
        retriever.fit([])


def test_retrieve_before_fit():
    """Проверяет поиск до подготовки модели."""
    retriever = TfidfRetriever()

    with pytest.raises(RuntimeError):
        retriever.retrieve("машинное обучение")


def test_retrieve_empty_query(retriever):
    """Проверяет запрет на пустой поисковый запрос."""
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_invalid_top_k(retriever):
    """Проверяет запрет на неположительное количество результатов."""
    with pytest.raises(ValueError):
        retriever.retrieve("машинное обучение", top_k=0)


def test_retrieve_returns_results(retriever):
    """Проверяет структуру результатов поиска."""
    results = retriever.retrieve("машинное обучение")

    assert len(results) > 0
    assert all(isinstance(result, RetrievalResult) for result in results)

    for result in results:
        assert isinstance(result.index, int)
        assert isinstance(result.score, float)
        assert result.score > 0


def test_retrieve_ranking(retriever):
    """Проверяет порядок результатов по убыванию сходства."""
    results = retriever.retrieve("машинное обучение")

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_retrieve_top_k(retriever):
    """Проверяет ограничение количества результатов."""
    results = retriever.retrieve("машинное обучение", top_k=1)

    assert len(results) == 1


def test_retrieve_indices(retriever):
    """Проверяет корректность индексов найденных документов."""
    results = retriever.retrieve("Python программирование")

    assert results[0].index == 2


def test_retrieve_duplicate_documents():
    """Проверяет сохранение индексов одинаковых документов."""
    retriever = TfidfRetriever()

    retriever.fit([
        "Машинное обучение изучает алгоритмы.",
        "Python используется для программирования.",
        "Машинное обучение изучает алгоритмы."
    ])

    results = retriever.retrieve(
        "машинное обучение",
        top_k=3
    )

    indices = {result.index for result in results}

    assert 0 in indices
    assert 2 in indices