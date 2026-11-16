from dataclasses import dataclass


@dataclass
class Claim:
    """Представляет отдельное утверждение в документе."""

    text: str
    index: int
    sentence_index: int
    paragraph_index: int