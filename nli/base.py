from abc import ABC, abstractmethod

from models.nli_result import NLIResult


class NLIModel(ABC):
    """Базовый интерфейс для моделей Natural Language Inference."""

    @abstractmethod
    def predict(
        self,
        premise: str,
        hypothesis: str
    ) -> NLIResult:
        """
        Определяет логическую связь между двумя предложениями.

        premise — исходный текст;
        hypothesis — проверяемое утверждение.
        """
        pass