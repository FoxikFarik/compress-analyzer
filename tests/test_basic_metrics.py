import pytest

from models.paragraph import Paragraph
from models.sentence import Sentence
from models.document import Document
from metrics.basic import calculate_compression_ratio, calculate_duplicate_rate

def create_document_with_sentences(texts: list[str]) -> Document:
    """Создаёт документ с заданным набором предложений."""
    sentences = [
        Sentence(
            text=text,
            index=index,
            paragraph_index=0
        )
        for index, text in enumerate(texts)
    ]

    paragraph = Paragraph(
        text=" ".join(texts),
        index=0,
        sentences=sentences
    )

    return Document(
        name="summary.txt",
        text=" ".join(texts),
        paragraphs=[paragraph]
    )

def test_compression_ratio():
    source = Document(
        name="source.txt",
        text=" ".join(["слово"] * 100)
    )

    summary = Document(
        name="summary.txt",
        text=" ".join(["слово"] * 20)
    )

    assert calculate_compression_ratio(source, summary) == 0.2


def test_compression_ratio_empty_source():
    source = Document(name="source.txt", text="")
    summary = Document(name="summary.txt", text="Текст конспекта")

    with pytest.raises(ValueError):
        calculate_compression_ratio(source, summary)
        
def test_duplicate_rate_without_duplicates():
    summary = create_document_with_sentences([
        "Первое предложение.",
        "Второе предложение.",
        "Третье предложение."
    ])

    assert calculate_duplicate_rate(summary) == 0.0


def test_duplicate_rate_with_duplicates():
    summary = create_document_with_sentences([
        "Первое предложение.",
        "Второе предложение.",
        "Первое предложение."
    ])

    assert calculate_duplicate_rate(summary) == pytest.approx(1 / 3)


def test_duplicate_rate_multiple_duplicates():
    summary = create_document_with_sentences([
        "Повтор.",
        "Повтор.",
        "Повтор."
    ])

    assert calculate_duplicate_rate(summary) == pytest.approx(2 / 3)


def test_duplicate_rate_ignores_case_and_whitespace():
    summary = create_document_with_sentences([
        "Машинное обучение позволяет учиться.",
        "  МАШИННОЕ   ОБУЧЕНИЕ ПОЗВОЛЯЕТ УЧИТЬСЯ. "
    ])

    assert calculate_duplicate_rate(summary) == 0.5


def test_duplicate_rate_empty_summary():
    summary = Document(
        name="summary.txt",
        text="",
        paragraphs=[]
    )

    with pytest.raises(ValueError):
        calculate_duplicate_rate(summary)