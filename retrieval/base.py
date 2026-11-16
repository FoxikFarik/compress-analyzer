from abc import ABC, abstractmethod

from models.retrieval_result import RetrievalResult


class FragmentRetriever(ABC):
    """Базовый интерфейс для поиска фрагментов документа."""

    @abstractmethod
    def fit(self, documents: list[str]) -> None:
        """Подготавливает поисковую модель на исходных фрагментах."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """Возвращает индексы наиболее релевантных фрагментов."""
        pass
