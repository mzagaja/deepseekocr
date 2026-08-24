# deepseek-ocr-pdf

Replace a PDF's text layer with DeepSeek-OCR-2 output, positioned where the
text actually appears on the page. Everything runs locally through Ollama.

## Install

```bash
brew install ocrmypdf qpdf          # ghostscript and tesseract come along
ollama pull hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0
uv venv && uv pip install -e ".[dev]"
```

## Use

```bash
deepseek-ocr-pdf scanned.pdf searchable.pdf
```

Defaults to `--force-ocr`: every page is rasterized and its text layer rebuilt.
To keep genuine born-digital text and replace only machine-OCR'd text, pass
`--redo-ocr`. To OCR only pages with no text at all, pass `--skip-text`.

Unrecognized options go straight to ocrmypdf, so `--sidecar out.txt`,
`--jobs 4`, and `--deskew` work as usual.

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0` | Ollama model tag |
| `--ollama-host` | `http://localhost:11434` | Ollama base URL |
| `--timeout` | `300` | Seconds allowed per page |
| `--max-dim` | none | Shrink pages to this longest edge before OCR |
| `--no-coverage-guard` | off | Skip the dropped-text repair pass |

Leave `--max-dim` alone unless pages are so large they exhaust memory.
Shrinking costs accuracy for nothing: at 1024px the model merges several lines
into one box, at 1700px it returns one box per line, and both take the same
time because it retiles internally either way.

## How it works

ocrmypdf rasterizes each page and applies the existing-text policy. The plugin
sends the raster to Ollama with a `<|grounding|>` prompt and parses the labelled
boxes it returns. Tesseract then runs as a *detector only* — its text is thrown
away — to find any text the model missed. Missed regions are re-read as tight
crops and merged. The result becomes an `OcrElement` tree, which ocrmypdf's
fpdf2 renderer writes as an invisible text layer.

## Why the coverage guard exists

DeepSeek-OCR-2 stops early when it reaches an isolated element after a large
blank region, silently omitting it. Measured on 2026-08-24: a footer at 86% page
height was dropped on every full-page run at both Q4_K_M and Q8_0, with
`done_reason: stop` and no truncation — yet read perfectly from a tight crop.
For a tool that replaces a text layer, silently losing a line is the worst
failure mode, because nothing about the output looks wrong.

`tests/test_integration.py` pins that exact page.

## Known limits

- `--force-ocr` rasterizes born-digital pages: bigger files, no vector text.
- Word boxes are synthesized from line boxes by character count, so selection
  within a line is approximate.
- The guard only catches text Tesseract can find. Text both engines miss is
  still lost.
- The Q8_0 GGUF is a community conversion, not an official DeepSeek release.
- Roughly 8–10s per page, plus a second pass on pages needing repair.

## Tests

```bash
.venv/bin/pytest -m "not slow"   # fast, no Ollama needed
.venv/bin/pytest -m slow         # end-to-end, needs Ollama and the model
```
