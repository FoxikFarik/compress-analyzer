from models.document import Document
from models.fragment import Fragment
from retrieval.tfidf import TfidfRetriever


class FragmentSearch:
    """Поиск релевантных фрагментов в исходном документе."""

    def __init__(self, context_window: int = 1):
        if context_window < 0:
            raise ValueError("Размер контекста не может быть отрицательным")

        self.context_window = context_window
        self.retriever = TfidfRetriever()
        self.document: Document | None = None

        self.sentences = []
        self.paragraphs = []

    def fit(self, document: Document) -> None:
        """Подготавливает предложения исходного документа для поиска."""
        self.document = document
        self.sentences = []
        self.paragraphs = []

        for paragraph in document.paragraphs:
            for local_index, sentence in enumerate(paragraph.sentences):
                self.sentences.append(sentence.text)
                self.paragraphs.append(
                    (paragraph, local_index)
                )

        if not self.sentences:
            raise ValueError("Исходный документ не содержит предложений")

        self.retriever.fit(self.sentences)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[Fragment]:
        """Находит релевантные фрагменты с учётом контекста."""
        if self.document is None:
            raise RuntimeError("Поисковая модель не подготовлена")

        results = self.retriever.retrieve(query, top_k)
        fragments = []

        for result in results:
            paragraph, local_index = self.paragraphs[result.index]
            paragraph_sentences = paragraph.sentences

            start = max(
                0,
                local_index - self.context_window
            )

            end = min(
                len(paragraph_sentences),
                local_index + self.context_window + 1
            )

            selected = paragraph_sentences[start:end]

            fragments.append(
                Fragment(
                    text=" ".join(
                        sentence.text for sentence in selected
                    ),
                    paragraph_index=paragraph.index,
                    start_sentence_index=selected[0].index,
                    end_sentence_index=selected[-1].index,
                    score=result.score
                )
            )

        return fragments
