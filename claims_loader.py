import json
from pathlib import Path

from models.claim import Claim


def load_claims(path: str) -> tuple[list[Claim], list[Claim]]:
    """Загружает утверждения конспекта и эталона из JSON."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл с утверждениями не найден: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Корневой элемент JSON должен быть объектом")

    def parse_claims(key: str) -> list[Claim]:
        """Проверяет и преобразует список утверждений."""

        if key not in data:
            raise ValueError(
                f"В JSON отсутствует обязательный ключ: {key}"
            )

        items = data[key]

        if not isinstance(items, list):
            raise ValueError(
                f"Значение {key} должно быть списком"
            )

        claims = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Утверждение {index} в {key} должно быть объектом"
                )

            required_fields = (
                "text",
                "sentence_index",
                "paragraph_index"
            )

            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Утверждение {index} в {key}: "
                        f"отсутствует поле {field}"
                    )

            if not isinstance(item["text"], str):
                raise ValueError(
                    f"Поле text в утверждении {index} должно быть строкой"
                )

            if not isinstance(item["sentence_index"], int):
                raise ValueError(
                    f"Поле sentence_index в утверждении {index} "
                    "должно быть целым числом"
                )

            if not isinstance(item["paragraph_index"], int):
                raise ValueError(
                    f"Поле paragraph_index в утверждении {index} "
                    "должно быть целым числом"
                )
            
            if not item["text"].strip():
                raise ValueError(
                    f"Текст утверждения {index} в {key} не должен быть пустым"
                )

            if (
                type(item["sentence_index"]) is not int
                or item["sentence_index"] < 0
            ):
                raise ValueError(
                    f"Поле sentence_index в утверждении {index} "
                    "должно быть неотрицательным целым числом"
                )

            if (
                type(item["paragraph_index"]) is not int
                or item["paragraph_index"] < 0
            ):
                raise ValueError(
                    f"Поле paragraph_index в утверждении {index} "
                    "должно быть неотрицательным целым числом"
                )

            claims.append(
                Claim(
                    text=item["text"],
                    index=index,
                    sentence_index=item["sentence_index"],
                    paragraph_index=item["paragraph_index"]
                )
            )

        return claims

    summary_claims = parse_claims("summary_claims")
    reference_claims = parse_claims("reference_claims")

    return summary_claims, reference_claims
