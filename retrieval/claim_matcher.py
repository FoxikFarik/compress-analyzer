from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.claim import Claim
from models.claim_match_result import ClaimMatchResult
import numpy as np


class ClaimMatcher:
    """Сопоставление утверждений конспекта с эталонными."""

    def __init__(self, threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("Порог сходства должен быть от 0 до 1")

        self.threshold = threshold

    def match(
        self,
        summary_claims: list[Claim],
        reference_claims: list[Claim]
    ) -> list[ClaimMatchResult]:
        """Сопоставляет утверждения с уникальными парами."""

        if not summary_claims:
            return []

        if not reference_claims:
            return [
                ClaimMatchResult(
                    summary_claim=claim,
                    reference_claim=None,
                    similarity=0.0,
                    matched=False
                )
                for claim in summary_claims
            ]

        texts = [
            claim.text
            for claim in reference_claims + summary_claims
        ]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )

        analyzer = vectorizer.build_analyzer()

        # Проверяем, есть ли хотя бы один подходящий токен.
        if not any(analyzer(text) for text in texts):
            return [
                ClaimMatchResult(
                    summary_claim=claim,
                    reference_claim=None,
                    similarity=0.0,
                    matched=False
                )
                for claim in summary_claims
            ]

        vectors = vectorizer.fit_transform(texts)

        reference_vectors = vectors[:len(reference_claims)]
        summary_vectors = vectors[len(reference_claims):]

        similarities = cosine_similarity(
            summary_vectors,
            reference_vectors
        )

        # Добавляем фиктивные столбцы для несопоставленных утверждений.
        scores = similarities.copy()
        scores[scores < self.threshold] = 0.0

        scores = np.column_stack((
            scores,
            np.zeros((len(summary_claims), len(summary_claims)))
        ))

        # Находим глобально оптимальное распределение пар.
        summary_indices, assigned_indices = linear_sum_assignment(
            scores,
            maximize=True
        )

        assignments = dict(
            zip(summary_indices, assigned_indices)
        )

        results = []

        for summary_index, summary_claim in enumerate(summary_claims):
            reference_index = assignments[summary_index]

            if (
                reference_index < len(reference_claims)
                and similarities[summary_index][reference_index]
                >= self.threshold
            ):
                results.append(
                    ClaimMatchResult(
                        summary_claim=summary_claim,
                        reference_claim=reference_claims[reference_index],
                        similarity=float(
                            similarities[summary_index][reference_index]
                        ),
                        matched=True
                    )
                )
            else:
                results.append(
                    ClaimMatchResult(
                        summary_claim=summary_claim,
                        reference_claim=None,
                        similarity=0.0,
                        matched=False
                    )
                )

        return results