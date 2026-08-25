# deepseek-ocr-pdf

Replace a PDF's text layer with DeepSeek-OCR-2 output, positioned where the
text actually appears on the page. Everything runs locally through Ollama.

## Install

```bash
brew install ocrmypdf qpdf          # ghostscript and tesseract come along
ollama pull hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0

# venv for development and tests
uv venv && uv pip install -e ".[dev]"
# then either activate it or use uv run:
source .venv/bin/activate
# or
uv run deepseek-ocr-pdf --help

# global CLI (puts deepseek-ocr-pdf on your PATH via ~/.local/bin)
uv tool install -e .
```

## Use

```bash
deepseek-ocr-pdf scanned.pdf searchable.pdf
```

Defaults to `--redo-ocr`: any existing invisible text layer is stripped and replaced, but the original page image is preserved losslessly (verified: image JPEG MD5 identical before/after). To rasterize every page and discard all text (e.g. to fix vector-text PDFs), pass `--force-ocr`. To OCR only pages with no text at all, pass `--skip-text`.

Also defaults to `--output-type pdf`, which rewrites nothing but the text layer. ocrmypdf's own default is `--output-type auto`: it attempts a declarations-only PDF/A conversion with pikepdf, validates it with veraPDF, and falls back to Ghostscript when veraPDF is missing or the file does not validate.

That Ghostscript pass costs less image quality than it appears to. With ocrmypdf's settings (`LeaveColorUnchanged`, `auto` compression), JPEG streams pass through byte for byte, and a Flate bilevel image re-encoded to CCITT G4 came back pixel-identical. What it does change is `/Interpolate`, which PDF/A forbids and Ghostscript strips from every image — scanners including the ScanSnap set it, and dropping it makes some viewers render a zoomed scan more harshly. It also rebuilds the document structure around the images.

So the choice here is conservatism, not rescue: a tool that replaces an invisible text layer should not also rewrite the file. Archival output stays one flag away — pass `--output-type pdfa` (or `auto`, `pdfa-1`, `pdfa-2`, `pdfa-3`).

Unrecognized options go straight to ocrmypdf, so `--sidecar out.txt`,
`--jobs 4`, and `--deskew` work as usual.

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0` | Ollama model tag |
| `--ollama-host` | `http://localhost:11434` | Ollama base URL |
| `--timeout` | `300` | Seconds allowed per page |
| `--max-dim` | none | Shrink pages to this longest edge before OCR |
| `--no-coverage-guard` | off | Skip the dropped-text repair pass |
| `--output-type` | `pdf` | Passed to ocrmypdf. `pdf` leaves the file untouched apart from the text layer; `pdfa` and friends rebuild it via Ghostscript and strip `/Interpolate` |

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

- `--force-ocr` rasterizes born-digital pages: bigger files, no vector text. Default `--redo-ocr` preserves the original image and only replaces the invisible layer; use `--force-ocr` only when vector text must be flattened.
- Asking for PDF/A output hands the file to Ghostscript, which rebuilds it and strips `/Interpolate`. Image data survives, but the file is no longer byte-comparable to the input. Installing veraPDF lets ocrmypdf try its pikepdf path instead, which only adds an sRGB OutputIntent and XMP metadata — but that path is only used when the result validates, and a scan carrying `/Interpolate true` will not.
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
