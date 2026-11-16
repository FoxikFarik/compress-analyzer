import pytest

from models.document import Document
from models.paragraph import Paragraph
from models.sentence import Sentence
from retrieval.search import FragmentSearch


def create_document(paragraph_texts: list[list[str]]) -> Document:
    """Создаёт тестовый документ из списка абзацев."""
    paragraphs = []
    sentence_index = 0

    for paragraph_index, texts in enumerate(paragraph_texts):
        sentences = []

        for text in texts:
            sentences.append(
                Sentence(
                    text=text,
                    index=sentence_index,
                    paragraph_index=paragraph_index
                )
            )
            sentence_index += 1

        paragraphs.append(
            Paragraph(
                text=" ".join(texts),
                index=paragraph_index,
                sentences=sentences
            )
        )

    return Document(
        name="test.txt",
        text="\n\n".join(
            paragraph.text for paragraph in paragraphs
        ),
        paragraphs=paragraphs
    )


@pytest.fixture
def document():
    """Создаёт тестовый документ с двумя абзацами."""
    return create_document([
        [
            "Машинное обучение изучает алгоритмы.",
            "Нейронные сети являются частью машинного обучения.",
            "Python используется для программирования."
        ],
        [
            "Информационный поиск работает с документами.",
            "TF-IDF оценивает важность слов.",
            "Обратный индекс ускоряет поиск."
        ]
    ])


def test_retrieve_before_fit():
    """Проверяет поиск до подготовки модели."""
    search = FragmentSearch()

    with pytest.raises(RuntimeError):
        search.retrieve("машинное обучение")


def test_fit_empty_document():
    """Проверяет обработку документа без предложений."""
    search = FragmentSearch()
    empty_document = create_document([])

    with pytest.raises(ValueError):
        search.fit(empty_document)


def test_retrieve_finds_relevant_fragment(document):
    """Проверяет поиск релевантного предложения."""
    search = FragmentSearch(context_window=0)
    search.fit(document)

    results = search.retrieve("машинное обучение", top_k=1)

    assert len(results) == 1
    assert results[0].text == (
        "Машинное обучение изучает алгоритмы."
    )
    assert results[0].score > 0


def test_retrieve_adds_context(document):
    """Проверяет добавление соседних предложений."""
    search = FragmentSearch(context_window=1)
    search.fit(document)

    results = search.retrieve(
        "нейронные сети",
        top_k=1
    )

    assert len(results) == 1
    assert results[0].text == (
        "Машинное обучение изучает алгоритмы. "
        "Нейронные сети являются частью машинного обучения. "
        "Python используется для программирования."
    )


def test_context_does_not_cross_paragraph_boundary(document):
    """Проверяет, что контекст не выходит за границы абзаца."""
    search = FragmentSearch(context_window=1)
    search.fit(document)

    results = search.retrieve(
        "информационный поиск",
        top_k=1
    )

    assert len(results) == 1
    assert results[0].paragraph_index == 1
    assert results[0].text == (
        "Информационный поиск работает с документами. "
        "TF-IDF оценивает важность слов."
    )


def test_retrieve_duplicate_sentences():
    """Проверяет правильную обработку одинаковых предложений."""
    document = create_document([
        [
            "Начало первого абзаца.",
            "Общий термин встречается здесь.",
            "Конец первого абзаца."
        ],
        [
            "Начало второго абзаца.",
            "Общий термин встречается здесь.",
            "Конец второго абзаца."
        ]
    ])

    search = FragmentSearch(context_window=1)
    search.fit(document)

    results = search.retrieve(
        "общий термин",
        top_k=2
    )

    assert len(results) == 2

    fragments_by_paragraph = {
        fragment.paragraph_index: fragment.text
        for fragment in results
    }

    assert fragments_by_paragraph[0] == (
        "Начало первого абзаца. "
        "Общий термин встречается здесь. "
        "Конец первого абзаца."
    )

    assert fragments_by_paragraph[1] == (
        "Начало второго абзаца. "
        "Общий термин встречается здесь. "
        "Конец второго абзаца."
    )


def test_context_window_zero(document):
    """Проверяет поиск без добавления контекста."""
    search = FragmentSearch(context_window=0)
    search.fit(document)

    results = search.retrieve(
        "TF-IDF",
        top_k=1
    )

    assert len(results) == 1
    assert results[0].text == "TF-IDF оценивает важность слов."


def test_invalid_context_window():
    """Проверяет запрет на отрицательный размер контекста."""
    with pytest.raises(ValueError):
        FragmentSearch(context_window=-1)
