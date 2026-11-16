import pytest

from document_loader import load_document


def test_load_txt(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Hello world",
        encoding="utf-8"
    )

    document = load_document(str(file_path))

    assert document.text == "Hello world"
    assert document.name == "test.txt"


def test_load_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(str(file_path))


def test_load_unsupported_format(tmp_path):
    file_path = tmp_path / "test.pdf"
    file_path.write_text("Test", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(str(file_path))


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document("nonexistent.txt")
        
def test_load_txt_with_structure(tmp_path):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Первый абзац. Второе предложение.\n\n"
        "Второй абзац.",
        encoding="utf-8"
    )

    document = load_document(str(file_path))

    assert document.paragraph_count == 2
    assert document.sentence_count == 3

    assert document.paragraphs[0].index == 0
    assert document.paragraphs[1].index == 1

    assert document.paragraphs[0].sentences[0].text == "Первый абзац."