from nli.transformer import TransformerNLIModel


model = TransformerNLIModel()

examples = [
    # Перефразирования
    (
        "Вода замерзает при температуре 0 °C.",
        "Вода замерзает при нуле градусов."
    ),
    (
        "Вода замерзает при температуре 0 °C.",
        "Температура замерзания воды составляет ноль градусов."
    ),
    (
        "Вода замерзает при температуре 0 °C.",
        "При температуре 0 °C вода превращается в лёд."
    ),

    # Противоречия
    (
        "Вода замерзает при температуре 0 °C.",
        "Вода замерзает при температуре 100 °C."
    ),
    (
        "Вода замерзает при температуре 0 °C.",
        "Вода не замерзает при температуре 0 °C."
    ),

    # Нейтральные утверждения
    (
        "Вода замерзает при температуре 0 °C.",
        "Вода широко используется в промышленности."
    )
]

for premise, hypothesis in examples:
    result = model.predict(premise, hypothesis)

    print(f"\nИсходный текст: {premise}")
    print(f"Утверждение:    {hypothesis}")
    print(f"Entailment:     {result.entailment:.4f}")
    print(f"Contradiction:  {result.contradiction:.4f}")
    print(f"Neutral:        {result.neutral:.4f}")
    print(f"Класс:          {result.label}")