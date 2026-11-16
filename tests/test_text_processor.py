from text_processor import normalize_text, split_paragraphs


def test_normalize_text():
    text = "  Hello   world!  \n\n\nTest  "

    result = normalize_text(text)

    assert result == "Hello world!\n\nTest"


def test_normalize_empty_text():
    assert normalize_text("") == ""


def test_split_empty_text():
    assert split_paragraphs("") == []


def test_split_paragraphs():
    text = "Первый абзац. Второе предложение.\n\nВторой абзац."

    result = split_paragraphs(text)

    assert len(result) == 2
    assert len(result[0].sentences) == 2
    assert len(result[1].sentences) == 1


def test_sentence_indices():
    text = "Первое. Второе.\n\nТретье."

    result = split_paragraphs(text)

    sentences = [
        sentence
        for paragraph in result
        for sentence in paragraph.sentences
    ]

    assert [s.index for s in sentences] == [0, 1, 2]
    assert [s.paragraph_index for s in sentences] == [0, 0, 1]