import json

from claims_loader import load_claims
from document_loader import load_document

from metrics.evaluator import DocumentEvaluator
from metrics.faithfulness import FaithfulnessEvaluator
from metrics.reference import ReferenceMetricsEvaluator

from nli.fake import FakeNLIModel
from retrieval.search import FragmentSearch
from retrieval.claim_matcher import ClaimMatcher


def test_full_evaluation_pipeline(tmp_path):
    """Проверяет полный цикл оценки конспекта."""

    # Подготавливаем исходный документ и конспект.
    source_path = tmp_path / "source.txt"
    summary_path = tmp_path / "summary.txt"
    claims_path = tmp_path / "claims.json"

    source_path.write_text(
        "Вода замерзает при нуле градусов. "
        "Вода кипит при ста градусах.",
        encoding="utf-8"
    )

    summary_path.write_text(
        "Вода замерзает при нуле градусов. "
        "Вода кипит при ста градусах.",
        encoding="utf-8"
    )
    # Подготавливаем утверждения для сравнения с эталоном.
    claims_data = {
        "summary_claims": [
            {
                "text": "Вода замерзает при нуле градусов.",
                "sentence_index": 0,
                "paragraph_index": 0
            },
            {
                "text": "Вода кипит при ста градусах.",
                "sentence_index": 1,
                "paragraph_index": 0
            }
        ],
        "reference_claims": [
            {
                "text": "Вода замерзает при нуле градусов.",
                "sentence_index": 0,
                "paragraph_index": 0
            },
            {
                "text": "Вода кипит при ста градусах.",
                "sentence_index": 1,
                "paragraph_index": 0
            }
        ]
    }

    claims_path.write_text(
        json.dumps(claims_data, ensure_ascii=False),
        encoding="utf-8"
    )

    # Загружаем документы и утверждения.
    source = load_document(str(source_path))
    summary = load_document(str(summary_path))

    summary_claims, reference_claims = load_claims(
        str(claims_path)
    )
    
    from text_processor import split_paragraphs

    print("Разбиение исходного текста:")
    for paragraph in split_paragraphs(source.text):
        print("Абзац:", repr(paragraph.text))
        for sentence in paragraph.sentences:
            print("  Предложение:", repr(sentence.text))
            
    print(
        split_paragraphs(
            "Первое предложение. Второе предложение."
        )
    )

    # Настраиваем поиск и оценку Faithfulness.
    fragment_search = FragmentSearch()
    fragment_search.fit(source)

    faithfulness_evaluator = FaithfulnessEvaluator(
        fragment_search=fragment_search,
        nli_model=FakeNLIModel()
    )

    # Настраиваем сравнение с эталоном.
    reference_evaluator = ReferenceMetricsEvaluator(
        ClaimMatcher(threshold=0.5)
    )

    # Собираем общий оценщик.
    evaluator = DocumentEvaluator(
        faithfulness_evaluator=faithfulness_evaluator,
        reference_metrics_evaluator=reference_evaluator
    )

    # Выполняем полную оценку.
    report = evaluator.evaluate(
        source=source,
        summary=summary,
        summary_claims=summary_claims,
        reference_claims=reference_claims
    )

    print("Предложения источника:", source.sentence_count)
    print("Предложения конспекта:", summary.sentence_count)
    print("Результаты Faithfulness:", len(report.faithfulness.sentence_results))
    print("Результаты сопоставления:", len(report.reference_metrics.matches))

    # Проверяем базовые метрики.
    assert report.compression_ratio == 1.0
    assert report.duplicate_rate == 0.0

    # Проверяем Faithfulness.
    assert report.faithfulness is not None
    assert report.faithfulness.score == 0.9
    assert len(report.faithfulness.sentence_results) == 2

    # Проверяем Reference Metrics.
    assert report.reference_metrics is not None
    assert report.reference_metrics.precision == 1.0
    assert report.reference_metrics.recall == 1.0
    assert report.reference_metrics.f1 == 1.0
    assert len(report.reference_metrics.matches) == 2
