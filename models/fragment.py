from dataclasses import dataclass


@dataclass
class Fragment:
    """Представляет фрагмент исходного документа."""

    text: str
    paragraph_index: int
    start_sentence_index: int
    end_sentence_index: int
    score: float
