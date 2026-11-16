import argparse
from pathlib import Path

from document_loader import load_document

from metrics.evaluator import DocumentEvaluator
from metrics.faithfulness import FaithfulnessEvaluator
from metrics.reference import ReferenceMetricsEvaluator

from nli.transformer import TransformerNLIModel
from retrieval.search import FragmentSearch
from retrieval.claim_matcher import ClaimMatcher


def parse_arguments():
    """Обрабатывает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Оценка качества сокращённого документа."
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Путь к исходному документу."
    )

    parser.add_argument(
        "summary",
        type=Path,
        help="Путь к конспекту."
    )

    parser.add_argument(
        "claims",
        type=Path,
        nargs="?",
        help="Путь к файлу с утверждениями (необязательно)."
    )

    return parser.parse_args()


def create_evaluator(source):
    """Создаёт и настраивает общий оценщик документов."""

    fragment_search = FragmentSearch(context_window=1)
    fragment_search.fit(source)

    faithfulness_evaluator = FaithfulnessEvaluator(
        fragment_search=fragment_search,
        nli_model=TransformerNLIModel()
    )

    return DocumentEvaluator(
        faithfulness_evaluator=faithfulness_evaluator,
        reference_metrics_evaluator=ReferenceMetricsEvaluator(
            claim_matcher=ClaimMatcher(threshold=0.5)
        )
    )


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

        status = (
            "СОПОСТАВЛЕНО"
            if result.matched
            else "НЕ СОПОСТАВЛЕНО"
        )
        print(f"Статус:   {status}")

    print("-" * 60)


def main():
    """Запускает процесс оценки конспекта."""

    args = parse_arguments()

    for path in (args.source, args.summary):
        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден документ: {path}"
            )

    source = load_document(str(args.source))
    summary = load_document(str(args.summary))

    evaluator = create_evaluator(source)

    summary_claims = None
    reference_claims = None

    if args.claims is not None:
        if not args.claims.is_file():
            raise FileNotFoundError(
                f"Не найден файл с утверждениями: {args.claims}"
            )

        from claims_loader import load_claims

        summary_claims, reference_claims = load_claims(
            str(args.claims)
        )
    else:
        evaluator.reference_metrics_evaluator = None

    report = evaluator.evaluate(
        source=source,
        summary=summary,
        summary_claims=summary_claims,
        reference_claims=reference_claims
    )

    print_report(report, source, summary)


if __name__ == "__main__":
    main()
