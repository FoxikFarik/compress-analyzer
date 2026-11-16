import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from models.nli_result import NLIResult
from nli.base import NLIModel


class TransformerNLIModel(NLIModel):
    """NLI-модель на основе Hugging Face Transformers."""

    DEFAULT_MODEL = (
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

        self.tokenizer = None
        self.model = None

        self.device = torch.device("cpu")

    def _load_model(self) -> None:
        """Загружает токенизатор и модель."""

        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        premise: str,
        hypothesis: str
    ) -> NLIResult:
        """Определяет логическую связь между двумя предложениями."""

        if not premise.strip():
            raise ValueError("Исходный текст не должен быть пустым")

        if not hypothesis.strip():
            raise ValueError("Проверяемое утверждение не должно быть пустым")

        self._load_model()

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        labels = {
            label.lower(): index
            for index, label in self.model.config.id2label.items()
        }

        required_labels = {
            "entailment",
            "contradiction",
            "neutral"
        }

        if not required_labels.issubset(labels):
            raise ValueError(
                "Модель должна поддерживать три класса NLI: "
                "entailment, contradiction, neutral"
            )

        return NLIResult(
            entailment=float(probabilities[labels["entailment"]]),
            contradiction=float(probabilities[labels["contradiction"]]),
            neutral=float(probabilities[labels["neutral"]])
        )
