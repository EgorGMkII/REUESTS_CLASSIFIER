import pytest

from src.classification._llm import parse_json_response, response_text, valid_confidence


class Message:
    def __init__(self, content):
        self.content = content


def test_response_text_supports_common_content_shapes():
    assert response_text("text") == "text"
    assert response_text(Message("message")) == "message"
    assert response_text(Message([{"text": "one"}, " two"])) == "one two"
    assert '"code": "3"' in response_text({"code": "3"})
    assert response_text(object()) == ""


def test_parse_json_supports_fences_and_rejects_invalid_shapes():
    assert parse_json_response("```json\n{\"code\":\"3\"}\n```")["code"] == "3"
    assert parse_json_response({"code": "2"})["code"] == "2"
    with pytest.raises(ValueError, match="empty"):
        parse_json_response("")
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_response("[]")


@pytest.mark.parametrize("value", [True, -0.1, 1.1, "bad"])
def test_confidence_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        valid_confidence(value)


def test_confidence_accepts_numeric_string():
    assert valid_confidence("0.75") == 0.75
