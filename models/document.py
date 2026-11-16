from dataclasses import dataclass, field

from models.paragraph import Paragraph


@dataclass
class Document:
    """Представляет текстовый документ."""

    name: str
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        """Возвращает количество слов в документе."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Возвращает количество символов в документе."""
        return len(self.text)

    @property
    def sentence_count(self) -> int:
        """Возвращает количество предложений в документе."""
        return sum(
            len(paragraph.sentences)
            for paragraph in self.paragraphs
        )

    @property
    def paragraph_count(self) -> int:
        """Возвращает количество абзацев в документе."""
        return len(self.paragraphs)