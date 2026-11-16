from dataclasses import dataclass, field

from models.sentence import Sentence


@dataclass
class Paragraph:
    """Представляет абзац документа."""

    text: str
    index: int
    sentences: list[Sentence] = field(default_factory=list)