from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Представляет результат поиска фрагмента документа."""

    index: int
    score: float
