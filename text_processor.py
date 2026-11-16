import re

from razdel import sentenize

from models.sentence import Sentence
from models.paragraph import Paragraph


def normalize_text(text: str) -> str:
    """
    Нормализует пробельные символы в тексте.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_paragraphs(text: str) -> list[Paragraph]:
    """
    Разбивает текст на абзацы и содержащиеся в них предложения.
    """
    text = normalize_text(text)

    if not text:
        return []

    raw_paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = []
    sentence_index = 0

    for paragraph_index, paragraph_text in enumerate(raw_paragraphs):
        sentences = []

        for sentence in sentenize(paragraph_text):
            sentences.append(
                Sentence(
                    text=sentence.text,
                    index=sentence_index,
                    paragraph_index=paragraph_index
                )
            )

            sentence_index += 1

        paragraphs.append(
            Paragraph(
                text=paragraph_text,
                index=paragraph_index,
                sentences=sentences
            )
        )

    return paragraphs