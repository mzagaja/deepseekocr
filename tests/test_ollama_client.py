# tests/test_ollama_client.py
import pytest
from PIL import Image

from deepseek_ocr_pdf.ollama_client import (
    DEFAULT_MODEL,
    GROUNDING_PROMPT,
    OllamaClient,
    OllamaUnavailable,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(path)
    return path


def test_default_model_is_the_q8_conversion():
    assert DEFAULT_MODEL == "hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0"


def test_grounding_prompt_carries_the_special_token():
    assert GROUNDING_PROMPT.startswith("<image>\n<|grounding|>")


def test_request_payload_shape(monkeypatch, image):
    sent = {}

    def fake_post(self, payload):
        sent.update(payload)
        return {"response": "text[[0, 0, 9, 9]]\nhi"}

    monkeypatch.setattr(OllamaClient, "_post", fake_post)
    client = OllamaClient()
    assert client.ocr_image(image) == "text[[0, 0, 9, 9]]\nhi"
    assert sent["model"] == DEFAULT_MODEL
    assert sent["stream"] is False
    assert sent["prompt"] == GROUNDING_PROMPT
    assert len(sent["images"]) == 1


def test_empty_response_is_retried_once(monkeypatch, image):
    calls = []

    def fake_post(self, payload):
        calls.append(1)
        return {"response": "" if len(calls) == 1 else "text[[0, 0, 9, 9]]\nok"}

    monkeypatch.setattr(OllamaClient, "_post", fake_post)
    assert OllamaClient().ocr_image(image).endswith("ok")
    assert len(calls) == 2


def test_two_empty_responses_return_empty_not_raise(monkeypatch, image):
    monkeypatch.setattr(OllamaClient, "_post", lambda self, payload: {"response": ""})
    assert OllamaClient().ocr_image(image) == ""


def test_connection_error_raises_unavailable(monkeypatch, image):
    def boom(self, payload):
        raise OSError("connection refused")

    monkeypatch.setattr(OllamaClient, "_post", boom)
    with pytest.raises(OllamaUnavailable) as excinfo:
        OllamaClient().ocr_image(image)
    assert "ollama pull" in str(excinfo.value)


def test_no_resizing_by_default():
    original = Image.new("RGB", (2000, 3000))
    assert OllamaClient()._fit(original).size == (2000, 3000)


def test_max_dim_shrinks_and_keeps_aspect_ratio():
    original = Image.new("RGB", (2000, 4000))
    assert OllamaClient(max_dim=1000)._fit(original).size == (500, 1000)


def test_max_dim_does_not_upscale_small_images():
    original = Image.new("RGB", (100, 50))
    assert OllamaClient(max_dim=1000)._fit(original).size == (100, 50)
