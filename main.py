from pathlib import Path

from claims_loader import load_claims
from document_loader import load_document

from metrics.evaluator import DocumentEvaluator
from metrics.faithfulness import FaithfulnessEvaluator
from metrics.reference import ReferenceMetricsEvaluator

from models.claim import Claim

from nli.fake import FakeNLIModel
from retrieval.search import FragmentSearch
from retrieval.claim_matcher import ClaimMatcher


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

SOURCE_PATH = DOCUMENTS_DIR / "AI_Basics_original.docx"
SUMMARY_PATH = DOCUMENTS_DIR / "AI_Basics_compressed.docx"
CLAIMS_PATH = DOCUMENTS_DIR / "claims.json"


def print_report(report, source, summary):
    """Выводит результаты оценки конспекта."""

    print("\n========== ОЦЕНКА КОНСПЕКТА ==========\n")

    print("Документы:")
    print(f"  Исходный документ: {source.name}")
    print(f"  Конспект:          {summary.name}")

    print("\nРазмер документов:")
    print(f"  Слов в источнике:  {source.word_count}")
    print(f"  Слов в конспекте:  {summary.word_count}")

    print("\nМетрики:")

    print(f"  Compression Ratio: {report.compression_ratio:.2%}")
    print(f"  Duplicate Rate:    {report.duplicate_rate:.2%}")

    if report.faithfulness is not None:
        faithfulness = report.faithfulness

        if faithfulness.score is not None:
            print(f"  Faithfulness:      {faithfulness.score:.2%}")

        print("\nПроверка предложений:")

        for result in faithfulness.sentence_results:
            print(f"\n  Предложение: {result.sentence}")
            print(f"  Entailment:   {result.score:.2%}")

    if report.reference_metrics is not None:
        metrics = report.reference_metrics

        print("\nСравнение с эталоном:")
        print(f"  Precision: {metrics.precision:.2%}")
        print(f"  Recall:    {metrics.recall:.2%}")
        print(f"  F1:        {metrics.f1:.2%}")

        print_claim_matches(metrics.matches)

    print("\n======================================\n")


def print_claim_matches(matches):
    """Выводит подробные результаты сопоставления утверждений."""

    print("\nПодробное сопоставление утверждений:")
    print("-" * 60)

    for result in matches:
        print(f"\nКонспект: {result.summary_claim.text}")

        if result.reference_claim is not None:
            print(f"Эталон:   {result.reference_claim.text}")
        else:
            print("Эталон:   не найден")

        print(f"Сходство: {result.similarity:.2%}")

        status = "СОПОСТАВЛЕНО" if result.matched else "НЕ СОПОСТАВЛЕНО"
        print(f"Статус:   {status}")

    print("-" * 60)



def main():
    """Запускает процесс оценки конспекта."""

    source = load_document(str(SOURCE_PATH))
    summary = load_document(str(SUMMARY_PATH))

    summary_claims, reference_claims = load_claims(
        str(CLAIMS_PATH)
    )

    # Подготавливаем поиск по исходному документу.
    fragment_search = FragmentSearch(context_window=1)
    fragment_search.fit(source)

    # Создаём оценщик Faithfulness.
    faithfulness_evaluator = FaithfulnessEvaluator(
        fragment_search=fragment_search,
        nli_model=FakeNLIModel()
    )

    # Создаём оценщик Reference Metrics.
    reference_metrics_evaluator = ReferenceMetricsEvaluator(
        claim_matcher=ClaimMatcher(threshold=0.5)
    )

    # Создаём общий оценщик.
    evaluator = DocumentEvaluator(
        faithfulness_evaluator=faithfulness_evaluator,
        reference_metrics_evaluator=reference_metrics_evaluator
    )

    report = evaluator.evaluate(
        source=source,
        summary=summary,
        summary_claims=summary_claims,
        reference_claims=reference_claims
    )

    print_report(report, source, summary)


if __name__ == "__main__":
    main()
