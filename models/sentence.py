from dataclasses import dataclass


@dataclass
class Sentence:
    """Представляет предложение документа."""

    text: str
    index: int
    paragraph_index: int