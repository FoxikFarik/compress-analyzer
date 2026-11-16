from unittest.mock import Mock

import pytest

from metrics.evaluator import DocumentEvaluator
from models.document import Document
from models.paragraph import Paragraph
from models.sentence import Sentence
from models.evaluation_report import EvaluationReport
from models.faithfulness_report import FaithfulnessReport
from metrics.reference import ReferenceMetricsEvaluator
from retrieval.claim_matcher import ClaimMatcher
from models.claim import Claim
from metrics.faithfulness import FaithfulnessEvaluator
from retrieval.search import FragmentSearch
from nli.base import NLIModel
from models.nli_result import NLIResult


@pytest.fixture
def documents():
    """Создаёт исходный документ и конспект."""

    source_text = "Одно два три четыре пять шесть"
    summary_text = "Одно два"

    source = Document(
        name="source.txt",
        text=source_text,
        paragraphs=[
            Paragraph(
                text=source_text,
                index=0,
                sentences=[
                    Sentence(
                        text=source_text,
                        index=0,
                        paragraph_index=0
                    )
                ]
            )
        ]
    )

    summary = Document(
        name="summary.txt",
        text=summary_text,
        paragraphs=[
            Paragraph(
                text=summary_text,
                index=0,
                sentences=[
                    Sentence(
                        text=summary_text,
                        index=0,
                        paragraph_index=0
                    )
                ]
            )
        ]
    )

    return source, summary


def test_evaluate_calculates_basic_metrics(documents):
    """Проверяет вычисление базовых метрик."""

    source, summary = documents

    evaluator = DocumentEvaluator()

    report = evaluator.evaluate(source, summary)

    assert isinstance(report, EvaluationReport)
    assert report.compression_ratio == pytest.approx(2 / 6)
    assert report.duplicate_rate == 0.0
    assert report.faithfulness is None


def test_evaluate_uses_faithfulness_evaluator(documents):
    """Проверяет вызов оценщика Faithfulness."""

    source, summary = documents

    faithfulness_evaluator = Mock()

    expected_report = FaithfulnessReport()

    faithfulness_evaluator.evaluate_report.return_value = (
        expected_report
    )

    evaluator = DocumentEvaluator(
        faithfulness_evaluator=faithfulness_evaluator
    )

    report = evaluator.evaluate(source, summary)

    faithfulness_evaluator.evaluate_report.assert_called_once_with(
        summary
    )

    assert report.faithfulness is expected_report


def test_evaluate_without_faithfulness(documents):
    """Проверяет работу без оценщика Faithfulness."""

    source, summary = documents

    evaluator = DocumentEvaluator()

    report = evaluator.evaluate(source, summary)

    assert report.faithfulness is None
    



def make_document(name, sentences):
    """Создаёт тестовый документ из списка предложений."""

    text = " ".join(sentences)

    paragraph = Paragraph(
        text=text,
        index=0,
        sentences=[
            Sentence(
                text=sentence,
                index=index,
                paragraph_index=0
            )
            for index, sentence in enumerate(sentences)
        ]
    )

    return Document(
        name=name,
        text=text,
        paragraphs=[paragraph]
    )

def test_evaluator_without_optional_components():
    source = make_document(
        "source",
        [
            "Вода замерзает при нуле градусов.",
            "Земля вращается вокруг Солнца."
        ]
    )

    summary = make_document(
        "summary",
        [
            "Вода замерзает при нуле градусов."
        ]
    )

    evaluator = DocumentEvaluator()

    report = evaluator.evaluate(source, summary)

    assert report.compression_ratio == 5 / 9
    assert report.duplicate_rate == 0.0

    assert report.faithfulness is None
    assert report.reference_metrics is None
    



def test_evaluator_with_reference_metrics():
    source = make_document(
        "source",
        ["Вода замерзает при нуле градусов."]
    )

    summary = make_document(
        "summary",
        ["Вода замерзает при нуле градусов."]
    )

    summary_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0)
    ]

    reference_claims = [
        Claim("Вода замерзает при нуле градусов.", 0, 0, 0)
    ]

    reference_evaluator = ReferenceMetricsEvaluator(
        ClaimMatcher()
    )

    evaluator = DocumentEvaluator(
        reference_metrics_evaluator=reference_evaluator
    )

    report = evaluator.evaluate(
        source,
        summary,
        summary_claims,
        reference_claims
    )

    assert report.reference_metrics is not None
    assert report.reference_metrics.precision == 1.0
    assert report.reference_metrics.recall == 1.0
    assert report.reference_metrics.f1 == 1.0

    assert report.faithfulness is None
    


def test_evaluator_requires_claims_for_reference_metrics():
    source = make_document(
        "source",
        ["Вода замерзает при нуле градусов."]
    )

    summary = make_document(
        "summary",
        ["Вода замерзает при нуле градусов."]
    )

    reference_evaluator = ReferenceMetricsEvaluator(
        ClaimMatcher()
    )

    evaluator = DocumentEvaluator(
        reference_metrics_evaluator=reference_evaluator
    )

    with pytest.raises(
        ValueError,
        match="необходимо передать утверждения"
    ):
        evaluator.evaluate(source, summary)
        
class FakeNLIModel(NLIModel):
    """Тестовая модель NLI."""

    def predict(self, premise, hypothesis):
        return NLIResult(
            entailment=0.9,
            contradiction=0.05,
            neutral=0.05
        )


def test_evaluator_with_faithfulness():
    source = make_document(
        "source",
        ["Вода замерзает при нуле градусов."]
    )

    summary = make_document(
        "summary",
        ["Вода замерзает при нуле градусов."]
    )

    fragment_search = FragmentSearch()
    fragment_search.fit(source)

    faithfulness_evaluator = FaithfulnessEvaluator(
        fragment_search=fragment_search,
        nli_model=FakeNLIModel()
    )

    evaluator = DocumentEvaluator(
        faithfulness_evaluator=faithfulness_evaluator
    )

    report = evaluator.evaluate(source, summary)

    assert report.faithfulness is not None
    assert len(report.faithfulness.sentence_results) == 1
    assert report.faithfulness.score == 0.9

    assert report.reference_metrics is None