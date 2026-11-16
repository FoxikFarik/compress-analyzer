from unittest.mock import Mock

import pytest

from metrics.faithfulness import FaithfulnessEvaluator
from models.faithfulness_report import FaithfulnessReport
from models.document import Document
from models.paragraph import Paragraph
from models.sentence import Sentence
from models.fragment import Fragment
from models.nli_result import NLIResult
from models.faithfulness_result import (
    FaithfulnessResult,
    FragmentCheckResult
)


class MockFragmentSearch:
    """Имитация поиска фрагментов документа."""

    def __init__(self, fragments):
        self.fragments = fragments
        self.queries = []

    def retrieve(self, query, top_k=3):
        self.queries.append((query, top_k))
        return self.fragments[:top_k]


class MockNLIModel:
    """Имитация модели Natural Language Inference."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.results[len(self.calls) - 1]

@pytest.fixture
def summary():
    """Создаёт тестовый конспект."""

    sentences = [
        Sentence(
            text="Машинное обучение использует данные.",
            index=0,
            paragraph_index=0
        ),
        Sentence(
            text="Модели обучаются на примерах.",
            index=1,
            paragraph_index=0
        )
    ]

    paragraph = Paragraph(
        text=" ".join(sentence.text for sentence in sentences),
        index=0,
        sentences=sentences
    )

    return Document(
        name="summary.txt",
        text=paragraph.text,
        paragraphs=[paragraph]
    )


@pytest.fixture
def fragments():
    """Создаёт тестовые фрагменты источника."""

    return [
        Fragment(
            text="Первый фрагмент источника.",
            paragraph_index=0,
            start_sentence_index=0,
            end_sentence_index=0,
            score=0.9
        ),
        Fragment(
            text="Второй фрагмент источника.",
            paragraph_index=0,
            start_sentence_index=1,
            end_sentence_index=1,
            score=0.8
        )
    ]

def test_evaluate_returns_result_for_each_sentence(summary, fragments):
    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.7, 0.2, 0.1),
        NLIResult(0.9, 0.05, 0.05),
        NLIResult(0.6, 0.2, 0.2)
    ])

    evaluator = FaithfulnessEvaluator(search, nli)

    results = evaluator.evaluate(summary)

    assert len(results) == 2
    assert all(isinstance(r, FaithfulnessResult) for r in results)
    
def test_evaluate_searches_each_sentence(summary, fragments):
    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.7, 0.2, 0.1),
        NLIResult(0.9, 0.05, 0.05),
        NLIResult(0.6, 0.2, 0.2)
    ])

    evaluator = FaithfulnessEvaluator(search, nli)

    evaluator.evaluate(summary)

    assert search.queries == [
        ("Машинное обучение использует данные.", 3),
        ("Модели обучаются на примерах.", 3)
    ]
    
def test_evaluate_passes_correct_text_to_nli(summary, fragments):
    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.7, 0.2, 0.1),
        NLIResult(0.9, 0.05, 0.05),
        NLIResult(0.6, 0.2, 0.2)
    ])

    evaluator = FaithfulnessEvaluator(search, nli)

    evaluator.evaluate(summary)

    assert nli.calls == [
        (fragments[0].text, summary.paragraphs[0].sentences[0].text),
        (fragments[1].text, summary.paragraphs[0].sentences[0].text),
        (fragments[0].text, summary.paragraphs[0].sentences[1].text),
        (fragments[1].text, summary.paragraphs[0].sentences[1].text)
    ]
    
def test_evaluate_respects_top_k(summary, fragments):
    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.9, 0.05, 0.05)
    ])

    evaluator = FaithfulnessEvaluator(
        search,
        nli,
        top_k=1
    )

    results = evaluator.evaluate(summary)

    assert all(len(r.fragment_results) == 1 for r in results)
    assert len(nli.calls) == 2
    
def test_evaluate_without_fragments(summary):
    search = MockFragmentSearch([])
    nli = MockNLIModel([])

    evaluator = FaithfulnessEvaluator(search, nli)

    results = evaluator.evaluate(summary)

    assert len(results) == 2
    assert all(r.score == 0.0 for r in results)
    assert nli.calls == []
    
def test_evaluate_empty_summary():
    summary = Document(
        name="summary.txt",
        text="",
        paragraphs=[]
    )

    search = MockFragmentSearch([])
    nli = MockNLIModel([])

    evaluator = FaithfulnessEvaluator(search, nli)

    results = evaluator.evaluate(summary)

    assert results == []
    assert search.queries == []
    assert nli.calls == []
    
@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_evaluator_rejects_invalid_top_k(top_k):
    search = MockFragmentSearch([])
    nli = MockNLIModel([])

    with pytest.raises(ValueError):
        FaithfulnessEvaluator(search, nli, top_k=top_k)
        
def test_evaluate_report_returns_report(summary, fragments):
    """Проверяет создание итогового отчёта."""

    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.7, 0.2, 0.1),
        NLIResult(0.9, 0.05, 0.05),
        NLIResult(0.6, 0.2, 0.2)
    ])

    evaluator = FaithfulnessEvaluator(search, nli)

    report = evaluator.evaluate_report(summary)

    assert isinstance(report, FaithfulnessReport)
    assert len(report.sentence_results) == 2


def test_evaluate_report_calculates_score(summary, fragments):
    """Проверяет вычисление итоговой оценки."""

    search = MockFragmentSearch(fragments)
    nli = MockNLIModel([
        NLIResult(0.8, 0.1, 0.1),
        NLIResult(0.7, 0.2, 0.1),
        NLIResult(0.9, 0.05, 0.05),
        NLIResult(0.6, 0.2, 0.2)
    ])

    evaluator = FaithfulnessEvaluator(search, nli)

    report = evaluator.evaluate_report(summary)

    assert report.score == pytest.approx(0.85)


def test_evaluate_report_with_empty_summary():
    """Проверяет создание отчёта для пустого конспекта."""

    summary = Document(
        name="summary.txt",
        text="",
        paragraphs=[]
    )

    search = MockFragmentSearch([])
    nli = MockNLIModel([])

    evaluator = FaithfulnessEvaluator(search, nli)

    report = evaluator.evaluate_report(summary)

    assert isinstance(report, FaithfulnessReport)
    assert report.sentence_results == []
    assert report.score is None
    
def create_summary():
    """Создаёт тестовый конспект из двух предложений."""

    sentences = [
        Sentence(
            text="Вода замерзает при 0 °C.",
            index=0,
            paragraph_index=0
        ),
        Sentence(
            text="Вода кипит при 100 °C.",
            index=1,
            paragraph_index=0
        )
    ]

    return Document(
        name="summary.txt",
        text="Вода замерзает при 0 °C. Вода кипит при 100 °C.",
        paragraphs=[
            Paragraph(
                text="Вода замерзает при 0 °C. Вода кипит при 100 °C.",
                index=0,
                sentences=sentences
            )
        ]
    )


def test_evaluate_checks_each_sentence():
    """Проверяет, что каждое предложение передаётся на проверку."""

    search = Mock()
    nli = Mock()

    search.retrieve.return_value = []
    nli.predict.return_value = NLIResult(
        entailment=0.9,
        contradiction=0.05,
        neutral=0.05
    )

    evaluator = FaithfulnessEvaluator(search, nli)
    results = evaluator.evaluate(create_summary())

    assert len(results) == 2
    assert search.retrieve.call_count == 2

    assert search.retrieve.call_args_list[0].args[0] == (
        "Вода замерзает при 0 °C."
    )
    assert search.retrieve.call_args_list[1].args[0] == (
        "Вода кипит при 100 °C."
    )


def test_evaluate_passes_fragment_to_nli():
    """Проверяет передачу найденного фрагмента в NLI."""

    fragment = Fragment(
        text="Вода замерзает при температуре 0 °C.",
        paragraph_index=0,
        start_sentence_index=0,
        end_sentence_index=0,
        score=0.8
    )

    search = Mock()
    search.retrieve.return_value = [fragment]

    nli = Mock()
    nli.predict.return_value = NLIResult(
        entailment=0.9,
        contradiction=0.05,
        neutral=0.05
    )

    evaluator = FaithfulnessEvaluator(search, nli)
    results = evaluator.evaluate(create_summary())

    assert nli.predict.call_count == 2

    nli.predict.assert_any_call(
        premise=fragment.text,
        hypothesis="Вода замерзает при 0 °C."
    )

    assert len(results[0].fragment_results) == 1


def test_evaluate_when_no_fragments_found():
    """Проверяет результат при отсутствии найденных фрагментов."""

    search = Mock()
    search.retrieve.return_value = []

    nli = Mock()

    evaluator = FaithfulnessEvaluator(search, nli)
    results = evaluator.evaluate(create_summary())

    assert len(results) == 2
    assert all(not result.fragment_results for result in results)
    assert all(result.score == 0.0 for result in results)

    nli.predict.assert_not_called()


def test_evaluate_empty_summary():
    """Проверяет обработку конспекта без предложений."""

    summary = Document(
        name="empty.txt",
        text="",
        paragraphs=[]
    )

    search = Mock()
    nli = Mock()

    evaluator = FaithfulnessEvaluator(search, nli)
    results = evaluator.evaluate(summary)

    assert results == []
    search.retrieve.assert_not_called()
    nli.predict.assert_not_called()