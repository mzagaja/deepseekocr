# src/deepseek_ocr_pdf/ollama_client.py
"""Talk to a local Ollama server.

Uses the native /api/generate endpoint rather than the OpenAI-compatible one,
which does not handle vision requests reliably.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0"
DEFAULT_TIMEOUT = 300.0

#: The trailing instruction matters. The model is sensitive to its prompt, and
#: this is the phrasing documented for grounding output.
GROUNDING_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable, or the model is not pulled."""


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_dim: int | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_dim = max_dim

    def _fit(self, image: Image.Image) -> Image.Image:
        """Shrink to max_dim, if one was set. Off by default.

        Coordinates are unaffected by resizing: the model normalizes each axis
        to 0-999 relative to whatever image it is sent, so a proportional
        resize cancels out when those values are rescaled to page pixels.

        Downscaling is off by default because it costs accuracy for nothing.
        Measured on 2026-08-24: at 1024px the model merges several lines into a
        single box; at 1700px it returns one box per line. Both took the same
        time (8.3s vs 8.4s) because the model retiles internally regardless.
        The flag exists only as an escape hatch for very large scans.
        """
        if not self.max_dim or max(image.size) <= self.max_dim:
            return image
        scale = self.max_dim / max(image.size)
        return image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.LANCZOS,
        )

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def ocr_image(self, image_path: Path, crop: tuple[int, int, int, int] | None = None) -> str:
        """OCR a page image, or a crop of it, and return the raw response.

        An empty response is retried once. A second empty response is returned
        as-is so that one unreadable page cannot kill a long document run.
        """
        with Image.open(image_path) as img:
            source = self._fit(img.crop(crop) if crop else img)
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")

        payload = {
            "model": self.model,
            "prompt": GROUNDING_PROMPT,
            "images": [base64.b64encode(buffer.getvalue()).decode()],
            "stream": False,
        }

        for attempt in (1, 2):
            try:
                text = (self._post(payload).get("response") or "").strip()
            except urllib.error.HTTPError as exc:
                raise OllamaUnavailable(
                    f"Ollama at {self.host} rejected the request ({exc.code}). "
                    f"Is the model pulled?  ollama pull {self.model}"
                ) from exc
            except OSError as exc:
                raise OllamaUnavailable(
                    f"Cannot reach Ollama at {self.host}: {exc}. "
                    f"Start it, then:  ollama pull {self.model}"
                ) from exc

            if text:
                return text
            if attempt == 1:
                log.warning("empty response from %s, retrying once", self.model)

        return ""
