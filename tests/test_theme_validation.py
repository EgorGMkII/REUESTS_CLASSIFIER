import pytest

from src.classifiers.theme_validation import (
    detect_suspicious_footnote_code,
    get_level,
    get_parent_code,
    is_working_classifier_code,
    normalize_name,
)


@pytest.mark.parametrize(
    ("code", "level"),
    [
        ("0005.0000.0000.0000", 1),
        ("0005.0005.0000.0000", 2),
        ("0005.0005.0056.0000", 3),
        ("0005.0005.0056.1149", 4),
    ],
)
def test_level(code, level):
    assert get_level(code) == level


def test_parent_code():
    assert get_parent_code("0005.0005.0056.1149") == "0005.0005.0056.0000"


def test_normalize_name():
    assert normalize_name("  Первая\n строка\xa0  текста ") == "Первая строка текста"


def test_methodical_code_is_not_working():
    assert not is_working_classifier_code("0000.0000.0000.1149")


def test_suspicious_footnote():
    result = detect_suspicious_footnote_code("0001.0001.0006.00161")
    assert result
    assert result["suggestedCode"] == "0001.0001.0006.0016"
