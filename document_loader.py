from pathlib import Path

from docx import Document as DocxDocument

from models.document import Document
from text_processor import split_paragraphs


def load_txt(path: Path) -> str:
    """Загружает текст из TXT-файла."""
    return path.read_text(encoding="utf-8")


def load_docx(path: Path) -> str:
    """Извлекает текст из DOCX-файла."""
    doc = DocxDocument(path)

    paragraphs = [
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]

    return "\n\n".join(paragraphs)


def load_document(path: str) -> Document:
    """Загружает документ и выполняет его структурирование."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    suffix = file_path.suffix.lower()

    loaders = {
        ".txt": load_txt,
        ".docx": load_docx,
    }

    loader = loaders.get(suffix)

    if loader is None:
        raise ValueError(
            f"Unsupported file format: {suffix}"
        )

    text = load_document_text(file_path, loader)

    if not text.strip():
        raise ValueError(
            f"Document is empty: {file_path}"
        )

    paragraphs = split_paragraphs(text)

    return Document(
        name=file_path.name,
        text=text,
        paragraphs=paragraphs
    )


def load_document_text(path: Path, loader) -> str:
    """Загружает и нормализует текст файла."""
    text = loader(path)
    return text