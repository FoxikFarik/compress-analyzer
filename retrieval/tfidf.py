from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.retrieval_result import RetrievalResult
from retrieval.base import FragmentRetriever


class TfidfRetriever(FragmentRetriever):
    """Поиск текстовых фрагментов с использованием TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[^\W\d_]+\b"
        )

        self.documents: list[str] = []
        self.document_vectors = None

    def fit(self, documents: list[str]) -> None:
        """Строит TF-IDF-векторы для исходных фрагментов."""
        if not documents:
            raise ValueError("Список документов пуст")

        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """Находит наиболее релевантные фрагменты."""
        if self.document_vectors is None:
            raise RuntimeError("Поисковая модель не обучена")

        if not query.strip():
            raise ValueError("Поисковый запрос пуст")

        if top_k <= 0:
            raise ValueError("top_k должен быть положительным")

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]

        ranked_indices = similarities.argsort()[::-1][:top_k]

        return [
            RetrievalResult(
                index=int(index),
                score=float(similarities[index])
            )
            for index in ranked_indices
            if similarities[index] > 0
        ]
