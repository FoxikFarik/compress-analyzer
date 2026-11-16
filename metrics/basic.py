
from models.document import Document


def calculate_compression_ratio(
    source: Document,
    summary: Document
) -> float:
    """Вычисляет отношение объёма конспекта к объёму источника."""
    if source.word_count == 0:
        raise ValueError("Исходный документ пуст")

    return summary.word_count / source.word_count


def calculate_duplicate_rate(summary: Document) -> float:
    """Вычисляет долю повторяющихся предложений в конспекте."""
    seen = set()
    sentence_count = 0
    duplicate_count = 0

    for paragraph in summary.paragraphs:
        for sentence in paragraph.sentences:
            normalized = " ".join(sentence.text.casefold().split())

            if not normalized:
                continue

            sentence_count += 1

            if normalized in seen:
                duplicate_count += 1
            else:
                seen.add(normalized)

    if sentence_count == 0:
        raise ValueError("Конспект не содержит предложений")

    return duplicate_count / sentence_count
