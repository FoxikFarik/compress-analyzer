import json
import pytest

from claims_loader import load_claims


def test_load_claims(tmp_path):
    data = {
        "summary_claims": [
            {
                "text": "Первое утверждение.",
                "sentence_index": 0,
                "paragraph_index": 0
            },
            {
                "text": "Второе утверждение.",
                "sentence_index": 1,
                "paragraph_index": 0
            }
        ],
        "reference_claims": [
            {
                "text": "Эталонное утверждение.",
                "sentence_index": 0,
                "paragraph_index": 0
            }
        ]
    }

    file_path = tmp_path / "claims.json"
    file_path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8"
    )

    summary_claims, reference_claims = load_claims(str(file_path))

    assert len(summary_claims) == 2
    assert len(reference_claims) == 1

    assert summary_claims[0].text == "Первое утверждение."
    assert summary_claims[0].index == 0
    assert summary_claims[1].index == 1

    assert reference_claims[0].text == "Эталонное утверждение."
    assert reference_claims[0].index == 0


def test_load_claims_with_empty_lists(tmp_path):
    file_path = tmp_path / "claims.json"

    file_path.write_text(
        json.dumps({
            "summary_claims": [],
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    summary_claims, reference_claims = load_claims(str(file_path))

    assert summary_claims == []
    assert reference_claims == []


def test_load_claims_missing_file(tmp_path):
    file_path = tmp_path / "missing.json"

    try:
        load_claims(str(file_path))
    except FileNotFoundError:
        pass
    else:
        assert False, "Ожидалась ошибка FileNotFoundError"

def test_load_claims_when_summary_claims_key_is_missing(tmp_path):
    """Проверяет обработку отсутствующего ключа summary_claims."""

    import json

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


def test_load_claims_when_reference_claims_key_is_missing(tmp_path):
    """Проверяет обработку отсутствующего ключа reference_claims."""

    import json

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


@pytest.mark.parametrize(
    "key",
    ["summary_claims", "reference_claims"]
)
def test_load_claims_when_claims_value_is_not_list(tmp_path, key):
    """Проверяет, что данные утверждений представлены списком."""

    import json

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": [],
            "reference_claims": [],
            key: "не список"
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


@pytest.mark.parametrize(
    "missing_field",
    ["text", "sentence_index", "paragraph_index"]
)
def test_load_claims_when_required_field_is_missing(
    tmp_path,
    missing_field
):
    """Проверяет обработку утверждения без обязательного поля."""

    import json

    claim = {
        "text": "Вода замерзает при 0 °C.",
        "sentence_index": 0,
        "paragraph_index": 0
    }

    del claim[missing_field]

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": [claim],
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))
        

def test_load_claims_with_empty_text(tmp_path):
    """Проверяет обработку пустого текста утверждения."""

    import json

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": [{
                "text": "",
                "sentence_index": 0,
                "paragraph_index": 0
            }],
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


@pytest.mark.parametrize(
    "field",
    ["sentence_index", "paragraph_index"]
)
def test_load_claims_with_negative_index(tmp_path, field):
    """Проверяет обработку отрицательных индексов."""

    import json

    claim = {
        "text": "Вода замерзает при 0 °C.",
        "sentence_index": 0,
        "paragraph_index": 0
    }

    claim[field] = -1

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": [claim],
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


@pytest.mark.parametrize(
    "field",
    ["sentence_index", "paragraph_index"]
)
def test_load_claims_with_boolean_index(tmp_path, field):
    """Проверяет, что логическое значение не принимается за индекс."""

    import json

    claim = {
        "text": "Вода замерзает при 0 °C.",
        "sentence_index": 0,
        "paragraph_index": 0
    }

    claim[field] = True

    path = tmp_path / "claims.json"
    path.write_text(
        json.dumps({
            "summary_claims": [claim],
            "reference_claims": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))


def test_load_claims_with_invalid_json(tmp_path):
    """Проверяет обработку некорректного JSON."""

    path = tmp_path / "claims.json"
    path.write_text(
        '{"summary_claims": [}',
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_claims(str(path))
