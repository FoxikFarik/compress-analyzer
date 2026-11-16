from models.document import Document
from models.paragraph import Paragraph
from models.sentence import Sentence


def test_word_count():
    document = Document(
        name="test.txt",
        text="Hello world from Python"
    )

    assert document.word_count == 4


def test_char_count():
    document = Document(
        name="test.txt",
        text="Hello"
    )

    assert document.char_count == 5


def test_empty_document():
    document = Document(
        name="empty.txt",
        text=""
    )

    assert document.word_count == 0
    assert document.char_count == 0
    assert document.sentence_count == 0
    assert document.paragraph_count == 0


def test_paragraph_and_sentence_count():
    document = Document(
        name="test.txt",
        text="Первое. Второе.\n\nТретье.",
        paragraphs=[
            Paragraph(
                text="Первое. Второе.",
                index=0,
                sentences=[
                    Sentence("Первое.", 0, 0),
                    Sentence("Второе.", 1, 0)
                ]
            ),
            Paragraph(
                text="Третье.",
                index=1,
                sentences=[
                    Sentence("Третье.", 2, 1)
                ]
            )
        ]
    )

    assert document.paragraph_count == 2
    assert document.sentence_count == 3


def test_default_paragraphs():
    document = Document(
        name="test.txt",
        text="Простой текст."
    )

    assert document.paragraphs == []