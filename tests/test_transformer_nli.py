from types import SimpleNamespace

import pytest
import torch

from nli.transformer import TransformerNLIModel


class FakeTokenizer:
    """Тестовый токенизатор."""

    def __call__(self, premise, hypothesis, **kwargs):
        self.premise = premise
        self.hypothesis = hypothesis

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }


class FakeModel:
    """Тестовая модель с заранее заданными оценками."""

    config = SimpleNamespace(
        id2label={
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
    )

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(
            logits=torch.tensor([[3.0, 1.0, 0.5]])
        )
    
    def to(self, device):
        return self

    def eval(self):
        return self


def test_predict_returns_nli_result(monkeypatch):
    """Проверяет получение результатов NLI."""

    import nli.transformer as transformer

    tokenizer = FakeTokenizer()

    monkeypatch.setattr(
        transformer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer
    )

    monkeypatch.setattr(
        transformer.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel()
    )

    model = TransformerNLIModel()

    result = model.predict(
        premise="Вода замерзает при 0 °C.",
        hypothesis="Вода замерзает при нуле градусов."
    )

    assert result.entailment > result.neutral
    assert result.entailment > result.contradiction

    assert result.entailment + result.neutral + result.contradiction == pytest.approx(1.0)


def test_predict_passes_texts_to_tokenizer(monkeypatch):
    """Проверяет передачу premise и hypothesis токенизатору."""

    import nli.transformer as transformer

    tokenizer = FakeTokenizer()

    monkeypatch.setattr(
        transformer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer
    )

    monkeypatch.setattr(
        transformer.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel()
    )

    model = TransformerNLIModel()

    model.predict(
        premise="Исходный текст.",
        hypothesis="Проверяемое утверждение."
    )

    assert tokenizer.premise == "Исходный текст."
    assert tokenizer.hypothesis == "Проверяемое утверждение."


@pytest.mark.parametrize(
    "premise, hypothesis",
    [
        ("", "Утверждение."),
        ("Исходный текст.", ""),
        ("   ", "Утверждение."),
        ("Исходный текст.", "   ")
    ]
)
def test_predict_rejects_empty_text(premise, hypothesis, monkeypatch):
    """Проверяет отклонение пустых входных текстов."""

    import nli.transformer as transformer

    monkeypatch.setattr(
        transformer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer()
    )

    monkeypatch.setattr(
        transformer.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel()
    )

    model = TransformerNLIModel()

    with pytest.raises(ValueError):
        model.predict(premise, hypothesis)
