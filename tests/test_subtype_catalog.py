import json

import pytest
from pydantic import ValidationError

from src.classification.subtype_catalog import (
    QuestionSubtypeCatalog,
    load_subtype_catalog,
)


def test_official_subtype_catalog_is_complete():
    catalog = load_subtype_catalog()
    assert len(catalog.items) == 39
    assert catalog.by_code()["1.3.4"].officialCode == "П.3.4"
    assert catalog.by_code()["3.2.4"].officialCode == "Ж2.4"
    assert catalog.allowed_codes()["9"] == ["-"]


def test_catalog_rejects_duplicate_codes():
    catalog = load_subtype_catalog().model_dump()
    catalog["items"].append(catalog["items"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        QuestionSubtypeCatalog.model_validate(catalog)
