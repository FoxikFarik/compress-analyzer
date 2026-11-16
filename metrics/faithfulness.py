from models.document import Document
from models.faithfulness_result import FaithfulnessResult
from models.faithfulness_report import FaithfulnessReport
from nli.base import NLIModel
from retrieval.search import FragmentSearch
from models.faithfulness_result import (
    FaithfulnessResult,
    FragmentCheckResult
)

class FaithfulnessEvaluator:
    """Оценка фактической согласованности конспекта с источником."""

    def __init__(
        self,
        fragment_search: FragmentSearch,
        nli_model: NLIModel,
        top_k: int = 3
    ):
        if top_k <= 0:
            raise ValueError("top_k должен быть положительным")

        self.fragment_search = fragment_search
        self.nli_model = nli_model
        self.top_k = top_k

    def evaluate(
        self,
        summary: Document
    ) -> list[FaithfulnessResult]:
        """Оценивает каждое предложение конспекта."""
        results = []

        for paragraph in summary.paragraphs:
            for sentence in paragraph.sentences:
                fragments = self.fragment_search.retrieve(
                    sentence.text,
                    top_k=self.top_k
                )

                fragment_results = []

                for fragment in fragments:
                    nli_result = self.nli_model.predict(
                        premise=fragment.text,
                        hypothesis=sentence.text
                    )

                    fragment_results.append(
                        FragmentCheckResult(
                            fragment=fragment,
                            nli_result=nli_result
                        )
                    )

                results.append(
                    FaithfulnessResult(
                        sentence=sentence.text,
                        fragment_results=fragment_results
                    )
                )

        return results
    
    def evaluate_report(
        self,
        summary: Document
    ) -> FaithfulnessReport:
        """Формирует итоговый отчёт об оценке Faithfulness."""

        results = self.evaluate(summary)

        return FaithfulnessReport(
            sentence_results=results
        )